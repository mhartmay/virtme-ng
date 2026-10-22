"""Tests for virtiofsd daemon management and virtio-fs QEMU export arguments."""

# The daemon handling is exercised against a real (fake) virtiofsd
# executable: what is checked is the observable behavior -- the command
# line the daemon records, the files it leaves behind, the messages
# printed to the user -- not how it is implemented.
# pylint: disable=protected-access

import contextlib
import io
import os
import shlex
import socket
import stat
import subprocess
import tempfile
import time
import unittest
from unittest import mock

import virtme_ng.virtiofs
from virtme_ng.virtiofs import (
    VirtioFS,
    VirtioFSConfig,
    VirtioFSState,
    ensure_virtiofs_memory,
    export_virtiofs,
)


def _rootfs_config(**kwargs):
    return VirtioFSConfig(path="/", mount_tag="ROOTFS", **kwargs)


class FakeArch:
    def __init__(self, virtiofs=True, numa=True):
        self._virtiofs = virtiofs
        self._numa = numa

    def virtiofs_support(self):
        return self._virtiofs

    def vhost_dev_type(self, _dev_type):
        return "virtiofs-device"

    def numa_support(self):
        return self._numa


@contextlib.contextmanager
def no_system_virtiofsd():
    """Make believe the machine has no virtiofsd outside the test's own.

    Determinism only: the fixed candidate paths (e.g. /usr/libexec/virtiofsd)
    may exist and work on the machine running the tests. Only used by tests
    that install no daemon of their own.
    """
    with (
        mock.patch("virtme_ng.virtiofs.which", return_value=None),
        mock.patch(
            "virtme_ng.virtiofs.subprocess.check_call",
            side_effect=subprocess.CalledProcessError(1, "virtiofsd"),
        ),
    ):
        yield


# What a started fake daemon does once it has recorded its command line:
# become ready (pid file + socket, like the real one) and serve, unless
# make_daemon() configures it otherwise.
_SERVER = """\
import os, socket, sys, time
args = sys.argv[1:]
sock = None
for i, a in enumerate(args):
    if a == "--socket-path" and i + 1 < len(args):
        sock = args[i + 1]
    elif a.startswith("--socket-path="):
        sock = a.split("=", 1)[1]
if {ready} and sock is not None:
    try:
        os.unlink(sock)
    except FileNotFoundError:
        pass
    with open(sock + ".pid", "w") as fd:
        fd.write(str(os.getpid()))
    socket.socket(socket.AF_UNIX, socket.SOCK_STREAM).bind(sock)
{extra}
for _ in range(60):
    time.sleep(1)
"""


class FakeDaemonTestCase(unittest.TestCase):
    """Base class for tests running against a fake virtiofsd executable."""

    def setUp(self):
        super().setUp()
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.guest_tools = self.tempdir.name
        self.daemon = os.path.join(self.guest_tools, "bin", "virtiofsd")
        self.make_daemon(self.daemon)
        # The --readonly support check caches its result globally so far:
        # reset it, or the first test's daemon would decide for all others.
        patcher = mock.patch.dict(
            virtme_ng.virtiofs._VIRTIOFSD_READONLY_SUPPORT, value=None
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_daemon(self, path, *, help_text="", role="serve", version_rc=0):
        """(Re-)install a fake virtiofsd at PATH.

        The daemon answers the --version and --help probes and records the
        command line of every real start in a log next to itself. ROLE is
        what a started daemon then does: "serve" (become ready like the
        real daemon and serve), "hang" (never become ready), "stubborn"
        (serve, but survive a SIGTERM), "bind-exit" (become ready, then
        exit) or "exit" (die right away).
        """
        extra = {
            "stubborn": "import signal\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)",
            "bind-exit": "sys.exit(3)",
        }.get(role, "")
        server = _SERVER.format(ready=role != "hang", extra=extra)
        script = f"""#!/bin/sh
for arg in "$@"; do
    case $arg in
    --version) exit {version_rc} ;;
    --help) printf %s {shlex.quote(help_text)}; exit 0 ;;
    esac
done
printf '%s\\0' '@@run@@' "$0" "$@" >> {shlex.quote(path + ".argv")}
"""
        if role == "exit":
            script += "exit 1\n"
        else:
            script += f'exec python3 -c {shlex.quote(server)} "$@"\n'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fd:
            fd.write(script)
        os.chmod(path, 0o755)

    def daemon_runs(self, path):
        """The command lines of the daemon's real starts, oldest first."""
        try:
            with open(path + ".argv", encoding="utf-8") as fd:
                records = fd.read().split("\0")
        except FileNotFoundError:
            return []
        runs, current = [], None
        for record in records:
            if record == "@@run@@":
                current = []
                runs.append(current)
            elif record and current is not None:
                current.append(record)
        return runs

    @staticmethod
    def flag_value(argv, flag):
        """The value FLAG is passed with in a recorded command line."""
        for i, arg in enumerate(argv):
            if arg == flag:
                return argv[i + 1]
            if arg.startswith(flag + "="):
                return arg.split("=", 1)[1]
        raise AssertionError(f"{flag} not found in {argv}")

    def start_daemon(self, config, verbose=False):
        """start() the daemon for CONFIG; it is cleaned up afterwards."""
        fs = VirtioFS(self.guest_tools)

        def cleanup():
            if fs.sock is not None:
                fs._cleanup_virtiofs_temp_files()

        self.addCleanup(cleanup)
        started = fs.start(
            config.path,
            verbose=verbose,
            cache="auto" if config.rw else "always",
            posix_acl=config.posix_acl,
            readonly=config.readonly,
        )
        return fs, started


class VirtioFSConfigTests(unittest.TestCase):
    def test_readonly_is_negated_rw(self):
        config = VirtioFSConfig(path="/", mount_tag="ROOTFS")
        self.assertFalse(config.rw)
        self.assertTrue(config.readonly)
        self.assertFalse(config.posix_acl)

        config = VirtioFSConfig(path="/", mount_tag="ROOTFS", rw=True)
        self.assertTrue(config.rw)
        self.assertFalse(config.readonly)

    def test_state_defaults(self):
        state = VirtioFSState("/opt/guest-tools")
        self.assertEqual(state.guest_tools_path, "/opt/guest-tools")
        self.assertIsNone(state.memory)
        self.assertFalse(state.memory_configured)


class EnsureVirtiofsMemoryTests(unittest.TestCase):
    def test_numa_appends_numa_node(self):
        qemuargs = []
        state = VirtioFSState("/opt/guest-tools")
        ensure_virtiofs_memory(FakeArch(numa=True), qemuargs, state)
        self.assertEqual(
            qemuargs,
            [
                "-object",
                "memory-backend-memfd,id=mem,size=128M,share=on",
                "-numa",
                "node,memdev=mem",
            ],
        )
        self.assertTrue(state.memory_configured)

    def test_without_numa_uses_machine_option(self):
        qemuargs = []
        state = VirtioFSState("/opt/guest-tools")
        ensure_virtiofs_memory(FakeArch(numa=False), qemuargs, state)
        self.assertEqual(
            qemuargs,
            [
                "-object",
                "memory-backend-memfd,id=mem,size=128M,share=on",
                "-machine",
                "memory-backend=mem",
            ],
        )

    def test_already_configured_is_noop(self):
        qemuargs = []
        state = VirtioFSState("/opt/guest-tools", memory_configured=True)
        ensure_virtiofs_memory(FakeArch(), qemuargs, state)
        self.assertEqual(qemuargs, [])

    def test_custom_memory_size(self):
        qemuargs = []
        state = VirtioFSState("/opt/guest-tools", memory="256M")
        ensure_virtiofs_memory(FakeArch(), qemuargs, state)
        self.assertIn("memory-backend-memfd,id=mem,size=256M,share=on", qemuargs)


class DaemonSelectionTests(FakeDaemonTestCase):
    def test_prefers_the_guest_tools_daemon(self):
        other = os.path.join(self.tempdir.name, "other", "virtiofsd")
        self.make_daemon(other)
        with mock.patch("virtme_ng.virtiofs.which", return_value=other):
            _, started = self.start_daemon(_rootfs_config())
        self.assertTrue(started)
        self.assertEqual(len(self.daemon_runs(self.daemon)), 1)
        self.assertEqual(self.daemon_runs(other), [])

    def test_skips_an_unusable_daemon(self):
        # The guest tools daemon is broken: the next candidate is used.
        self.make_daemon(self.daemon, version_rc=1)
        other = os.path.join(self.tempdir.name, "other", "virtiofsd")
        self.make_daemon(other)
        with mock.patch("virtme_ng.virtiofs.which", return_value=other):
            _, started = self.start_daemon(_rootfs_config())
        self.assertTrue(started)
        self.assertEqual(len(self.daemon_runs(other)), 1)
        self.assertEqual(self.daemon_runs(self.daemon), [])

    def test_falls_back_to_9p_without_a_usable_daemon(self):
        os.unlink(self.daemon)
        with no_system_virtiofsd():
            _, started = self.start_daemon(_rootfs_config())
        self.assertFalse(started)


class DaemonStartupTests(FakeDaemonTestCase):
    def test_passes_the_export_options_to_the_daemon(self):
        fs, started = self.start_daemon(_rootfs_config())
        self.assertTrue(started)
        (argv,) = self.daemon_runs(self.daemon)
        self.assertEqual(argv[0], self.daemon)
        self.assertIn("--syslog", argv)
        self.assertIn("--no-announce-submounts", argv)
        self.assertEqual(self.flag_value(argv, "--shared-dir"), "/")
        self.assertEqual(self.flag_value(argv, "--socket-path"), fs.sock)
        self.assertEqual(self.flag_value(argv, "--sandbox"), "none")
        self.assertEqual(self.flag_value(argv, "--cache"), "always")
        self.assertNotIn("--posix-acl", argv)
        self.assertNotIn("--readonly", argv)

    def test_cache_auto_for_rw_and_always_for_readonly(self):
        _, started = self.start_daemon(_rootfs_config(rw=True))
        self.assertTrue(started)
        argv = self.daemon_runs(self.daemon)[-1]
        self.assertEqual(self.flag_value(argv, "--cache"), "auto")

        _, started = self.start_daemon(_rootfs_config())
        self.assertTrue(started)
        argv = self.daemon_runs(self.daemon)[-1]
        self.assertEqual(self.flag_value(argv, "--cache"), "always")

    def test_posix_acl_when_requested(self):
        _, started = self.start_daemon(_rootfs_config(posix_acl=True))
        self.assertTrue(started)
        (argv,) = self.daemon_runs(self.daemon)
        self.assertIn("--posix-acl", argv)

    def test_readonly_when_advertised_by_the_daemon(self):
        self.make_daemon(self.daemon, help_text="--readonly")
        _, started = self.start_daemon(_rootfs_config())
        self.assertTrue(started)
        (argv,) = self.daemon_runs(self.daemon)
        self.assertIn("--readonly", argv)

    def test_readonly_dropped_when_not_advertised(self):
        _, started = self.start_daemon(_rootfs_config())
        self.assertTrue(started)
        (argv,) = self.daemon_runs(self.daemon)
        self.assertNotIn("--readonly", argv)

    def test_falls_back_to_9p_when_the_daemon_never_becomes_ready(self):
        self.make_daemon(self.daemon, role="hang")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            _, started = self.start_daemon(_rootfs_config(), verbose=True)
        self.assertFalse(started)
        self.assertIn("failed to start virtiofsd", stderr.getvalue())


class IsSocketTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)

    def test_true_for_unix_socket(self):
        path = os.path.join(self.tempdir.name, "sock")
        daemon_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        daemon_sock.bind(path)
        daemon_sock.close()
        self.assertTrue(VirtioFS._is_socket(path))

    def test_false_for_socket_of_another_user(self):
        # A local user could bind a socket of their own at the path; see the
        # docstring of _is_socket().
        foreign = mock.MagicMock(st_mode=stat.S_IFSOCK, st_uid=os.geteuid() + 1)
        with mock.patch("virtme_ng.virtiofs.os.stat", return_value=foreign):
            self.assertFalse(VirtioFS._is_socket("ignored"))

    def test_false_for_regular_file(self):
        path = os.path.join(self.tempdir.name, "file")
        with open(path, "w", encoding="utf-8") as fd:
            fd.write("")
        self.assertFalse(VirtioFS._is_socket(path))

    def test_false_for_missing_path(self):
        path = os.path.join(self.tempdir.name, "nonexistent")
        self.assertFalse(VirtioFS._is_socket(path))


class ExportVirtiofsTests(FakeDaemonTestCase):
    def test_without_virtiofs_support_falls_back(self):
        qemuargs = []
        state = VirtioFSState(self.guest_tools)
        self.assertFalse(
            export_virtiofs(FakeArch(virtiofs=False), qemuargs, state, _rootfs_config())
        )
        self.assertEqual(qemuargs, [])
        self.assertEqual(self.daemon_runs(self.daemon), [])

    def test_falls_back_to_9p_without_a_usable_daemon(self):
        os.unlink(self.daemon)
        qemuargs = []
        state = VirtioFSState(self.guest_tools)
        with no_system_virtiofsd():
            self.assertFalse(
                export_virtiofs(
                    FakeArch(), qemuargs, state, _rootfs_config(), verbose=True
                )
            )
        self.assertEqual(qemuargs, [])

    def test_exports_the_rootfs_over_the_daemon_socket(self):
        qemuargs = []
        state = VirtioFSState(self.guest_tools)
        self.assertTrue(export_virtiofs(FakeArch(), qemuargs, state, _rootfs_config()))
        chardev = next(
            arg for arg in qemuargs if arg.startswith("socket,id=charvirtfs0,path=")
        )
        # The daemon is serving at the socket QEMU is told to connect to.
        self.assertTrue(stat.S_ISSOCK(os.stat(chardev.split("path=", 1)[1]).st_mode))
        self.assertIn("virtiofs-device,chardev=charvirtfs0,tag=ROOTFS", qemuargs)
        self.assertIn("memory-backend-memfd,id=mem,size=128M,share=on", qemuargs)
        self.assertIn("-numa", qemuargs)
        self.assertTrue(state.memory_configured)


class CleanupTests(FakeDaemonTestCase):
    def test_stops_the_daemon_and_removes_its_temp_files(self):
        fs, started = self.start_daemon(_rootfs_config())
        self.assertTrue(started)
        with open(fs.pid, encoding="utf-8") as fd:
            pid = int(fd.read())

        fs._cleanup_virtiofs_temp_files()

        # The daemon is not the cleanup's child: give it a moment to die.
        for _ in range(100):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            self.fail("the daemon is still running")
        self.assertFalse(os.path.exists(fs.sock))
        self.assertFalse(os.path.exists(fs.pid))


if __name__ == "__main__":
    unittest.main()
