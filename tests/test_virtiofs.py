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
import unittest
from unittest import mock

from virtme.commands.run import (
    export_hostfs,
    guest_root_translation,
    make_parser,
    rootfs_ownership,
)
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

    def make_daemon(self, path, *, help_text="", help_rc=0, role="serve", version_rc=0):
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
            # Exit hard, without the interpreter finalization: the daemon
            # must be dead before its socket can be seen, or the readiness
            # check races with a still-dying process.
            "bind-exit": "os._exit(3)",
        }.get(role, "")
        server = _SERVER.format(ready=role != "hang", extra=extra)
        script = f"""#!/bin/sh
for arg in "$@"; do
    case $arg in
    --version) exit {version_rc} ;;
    --help) printf %s {shlex.quote(help_text)}; exit {help_rc} ;;
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

    def start_daemon(self, config, verbose=False, virtiofsd_path=None):
        """start() the daemon for CONFIG; it is cleaned up afterwards."""
        fs = VirtioFS(self.guest_tools, virtiofsd_path)

        def cleanup():
            if fs.sock is not None:
                fs.shutdown()

        self.addCleanup(cleanup)
        return fs, fs.start(config, verbose=verbose)


class VirtioFSConfigTests(unittest.TestCase):
    def test_readonly_is_negated_rw(self):
        config = VirtioFSConfig(path="/", mount_tag="ROOTFS")
        self.assertFalse(config.rw)
        self.assertTrue(config.readonly)
        self.assertFalse(config.posix_acl)
        self.assertIsNone(config.translate_uid)
        self.assertIsNone(config.translate_gid)

        config = VirtioFSConfig(path="/", mount_tag="ROOTFS", rw=True)
        self.assertTrue(config.rw)
        self.assertFalse(config.readonly)

    def test_zero_translate_id_is_normalized_to_none(self):
        # A 0 component is the identity: it must not become an instruction
        # to pass an identity map to virtiofsd.
        config = VirtioFSConfig(
            path="/", mount_tag="ROOTFS", translate_uid=0, translate_gid=1000
        )
        self.assertIsNone(config.translate_uid)
        self.assertEqual(config.translate_gid, 1000)
        config = VirtioFSConfig(
            path="/", mount_tag="ROOTFS", translate_uid=1000, translate_gid=0
        )
        self.assertEqual(config.translate_uid, 1000)
        self.assertIsNone(config.translate_gid)

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

    def test_reports_a_usable_daemon_without_starting_it(self):
        self.assertTrue(VirtioFS(self.guest_tools).is_virtiofsd_available())
        self.assertEqual(self.daemon_runs(self.daemon), [])

    def test_reports_no_usable_daemon(self):
        os.unlink(self.daemon)
        with no_system_virtiofsd():
            self.assertFalse(VirtioFS(self.guest_tools).is_virtiofsd_available())

    def test_uses_an_explicitly_requested_daemon(self):
        explicit = os.path.join(self.tempdir.name, "custom", "virtiofsd")
        self.make_daemon(explicit)
        _, started = self.start_daemon(_rootfs_config(), virtiofsd_path=explicit)
        self.assertTrue(started)
        (argv,) = self.daemon_runs(explicit)
        self.assertEqual(argv[0], explicit)
        # An explicitly requested daemon is the only candidate: the guest
        # tools one is silently installed, but never used.
        self.assertEqual(self.daemon_runs(self.daemon), [])

    def test_an_unusable_explicit_daemon_has_no_fallback(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            _, started = self.start_daemon(
                _rootfs_config(), virtiofsd_path="/nonexistent/virtiofsd"
            )
        self.assertFalse(started)
        self.assertEqual(self.daemon_runs(self.daemon), [])
        self.assertIn("is not a usable virtiofsd", stderr.getvalue())

    def test_an_empty_explicit_daemon_is_an_explicit_request(self):
        # An empty --virtiofsd (e.g. an unset shell variable) must not
        # silently re-enable the auto-detection.
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            _, started = self.start_daemon(_rootfs_config(), virtiofsd_path="")
        self.assertFalse(started)
        self.assertEqual(self.daemon_runs(self.daemon), [])
        self.assertIn("is not a usable virtiofsd", stderr.getvalue())

    def test_an_unusable_explicit_daemon_is_reported_only_once(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            for _ in range(2):
                _, started = self.start_daemon(
                    _rootfs_config(), virtiofsd_path="/nonexistent/virtiofsd"
                )
                self.assertFalse(started)
        self.assertEqual(stderr.getvalue().count("is not a usable virtiofsd"), 1)


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
        self.assertTrue(fs.sock.endswith(".sock"))
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

    def test_readonly_dropped_when_only_a_longer_flag_matches(self):
        # --readonly-mode is not --readonly: a prefix match would silently
        # pass an unsupported flag to the daemon.
        self.make_daemon(self.daemon, help_text="--readonly-mode")
        _, started = self.start_daemon(_rootfs_config())
        self.assertTrue(started)
        (argv,) = self.daemon_runs(self.daemon)
        self.assertNotIn("--readonly", argv)

    def test_flags_dropped_when_the_help_query_fails(self):
        self.make_daemon(self.daemon, help_text="--readonly", help_rc=1)
        _, started = self.start_daemon(_rootfs_config())
        self.assertTrue(started)
        (argv,) = self.daemon_runs(self.daemon)
        self.assertNotIn("--readonly", argv)

    def test_translates_the_guest_root_to_the_rootfs_owner(self):
        self.make_daemon(
            self.daemon, help_text="--translate-uid\n--translate-gid\n--readonly"
        )
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            _, started = self.start_daemon(
                _rootfs_config(
                    posix_acl=True,
                    translate_uid=1234,
                    translate_gid=5678,
                    translate_desc="the owner of the rootfs",
                )
            )
        self.assertTrue(started)
        (argv,) = self.daemon_runs(self.daemon)
        self.assertIn("--translate-uid=map:0:1234:1", argv)
        self.assertIn("--translate-gid=map:0:5678:1", argv)
        # UID/GID translation cannot be combined with POSIX ACLs.
        self.assertNotIn("--posix-acl", argv)
        self.assertIn(
            "mapping guest root to the owner of the rootfs (uid 1234, gid 5678)",
            stderr.getvalue(),
        )
        self.assertIn("disabling POSIX ACLs", stderr.getvalue())

    def test_the_mapping_target_is_named_without_a_description(self):
        # A VirtioFSConfig without translate_desc still announces the
        # mapping, naming the ids.
        self.make_daemon(self.daemon, help_text="--translate-uid\n--translate-gid")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            _, started = self.start_daemon(
                _rootfs_config(translate_uid=1000, translate_gid=1000)
            )
        self.assertTrue(started)
        self.assertIn("mapping guest root to uid 1000, gid 1000", stderr.getvalue())

    def test_without_translation_the_acls_are_kept(self):
        # What --no-root-uid-map asks for: the guest root is not mapped to
        # the rootfs owner, the POSIX ACLs are kept and nothing is announced.
        self.make_daemon(
            self.daemon, help_text="--translate-uid\n--translate-gid\n--readonly"
        )
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            _, started = self.start_daemon(_rootfs_config(posix_acl=True))
        self.assertTrue(started)
        (argv,) = self.daemon_runs(self.daemon)
        self.assertIn("--posix-acl", argv)
        self.assertFalse(any(arg.startswith("--translate") for arg in argv))
        self.assertEqual(stderr.getvalue(), "")

    def test_translate_dropped_when_not_advertised(self):
        # No translate flags in the --help output: the mapping is not even
        # attempted, the ACLs are kept and the loss is announced.
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            _, started = self.start_daemon(
                _rootfs_config(posix_acl=True, translate_uid=1000, translate_gid=1000)
            )
        self.assertTrue(started)
        (argv,) = self.daemon_runs(self.daemon)
        self.assertFalse(any(arg.startswith("--translate") for arg in argv))
        self.assertIn("--posix-acl", argv)
        self.assertIn(
            "does not support --translate-uid/--translate-gid", stderr.getvalue()
        )

    def test_missing_translate_flag_is_named_in_the_warning(self):
        self.make_daemon(self.daemon, help_text="--translate-uid")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            _, started = self.start_daemon(
                _rootfs_config(translate_uid=1000, translate_gid=1000)
            )
        self.assertTrue(started)
        (argv,) = self.daemon_runs(self.daemon)
        self.assertFalse(any(arg.startswith("--translate") for arg in argv))
        self.assertIn("does not support --translate-gid", stderr.getvalue())

    def test_lost_mapping_is_announced_with_the_9p_fallback(self):
        # A requested mapping that is lost with the fallback changes what
        # the guest sees: it is announced even when not verbose.
        self.make_daemon(
            self.daemon, help_text="--translate-uid\n--translate-gid", role="hang"
        )
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            _, started = self.start_daemon(
                _rootfs_config(translate_uid=1000, translate_gid=1000)
            )
        self.assertFalse(started)
        self.assertIn("fallback to 9p", stderr.getvalue())
        self.assertIn("not mapped to uid 1000, gid 1000", stderr.getvalue())

    def test_lost_mapping_is_announced_without_a_usable_daemon(self):
        # No usable virtiofsd at all: there is no daemon to probe for the
        # translate options, but the loss is announced anyway.
        os.unlink(self.daemon)
        stderr = io.StringIO()
        with no_system_virtiofsd(), contextlib.redirect_stderr(stderr):
            _, started = self.start_daemon(
                _rootfs_config(translate_uid=1000, translate_gid=1000)
            )
        self.assertFalse(started)
        self.assertIn("fallback to 9p", stderr.getvalue())
        self.assertIn("not mapped to uid 1000, gid 1000", stderr.getvalue())

    def test_falls_back_to_9p_when_the_daemon_never_becomes_ready(self):
        self.make_daemon(self.daemon, role="hang")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            _, started = self.start_daemon(_rootfs_config(), verbose=True)
        self.assertFalse(started)
        self.assertIn("failed to start virtiofsd", stderr.getvalue())

    def test_falls_back_to_9p_when_the_daemon_dies_before_the_socket(self):
        # An exited daemon must not be waited for until the timeout.
        self.make_daemon(self.daemon, role="exit")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            _, started = self.start_daemon(_rootfs_config(), verbose=True)
        self.assertFalse(started)
        self.assertIn("failed to start virtiofsd", stderr.getvalue())

    def test_falls_back_to_9p_when_the_daemon_dies_after_the_socket(self):
        # The socket path shows up at bind() time: a daemon that dies right
        # after creating it must not be mistaken for a ready one.
        self.make_daemon(self.daemon, role="bind-exit")
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

    def test_without_virtiofs_support_announces_the_lost_mapping(self):
        # start() is never reached without virtiofs support: a lost mapping
        # has to be announced by export_virtiofs() itself.
        qemuargs = []
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertFalse(
                export_virtiofs(
                    FakeArch(virtiofs=False),
                    qemuargs,
                    VirtioFSState(self.guest_tools),
                    _rootfs_config(translate_uid=1000, translate_gid=1000),
                )
            )
        self.assertEqual(qemuargs, [])
        self.assertIn("not mapped to uid 1000, gid 1000", stderr.getvalue())

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


class ExportHostfsTests(FakeDaemonTestCase):
    """The guest root mapping resolved for a shared dir reaches its daemon."""

    def _export(self, translate):
        self.make_daemon(
            self.daemon, help_text="--translate-uid\n--translate-gid\n--readonly"
        )
        qemuargs = []
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            fstype = export_hostfs(
                qemu=mock.Mock(),
                arch=FakeArch(),
                qemuargs=qemuargs,
                virtiofs_state=VirtioFSState(self.guest_tools),
                path="/home/user/temp",
                mount_tag="virtme.initmount0",
                readonly=False,
                translate=translate,
            )
        return fstype, stderr.getvalue()

    def test_the_translation_is_passed_to_the_daemon(self):
        fstype, stderr = self._export((1000, 1000, "the owner of the shared directory"))
        self.assertEqual(fstype, "virtiofs")
        (argv,) = self.daemon_runs(self.daemon)
        self.assertIn("--translate-uid=map:0:1000:1", argv)
        self.assertIn("--translate-gid=map:0:1000:1", argv)
        self.assertIn(
            "mapping guest root to the owner of the shared directory "
            "(uid 1000, gid 1000)",
            stderr,
        )

    def test_without_a_translation_no_flags_are_passed(self):
        # The internal exports (guest tools, modules, ...) share no guest
        # root mapping: nothing must be announced or passed to the daemon.
        fstype, stderr = self._export(None)
        self.assertEqual(fstype, "virtiofs")
        (argv,) = self.daemon_runs(self.daemon)
        self.assertFalse(any(arg.startswith("--translate") for arg in argv))
        self.assertEqual(stderr, "")


class RootfsOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)

    @staticmethod
    def _stat_as(uid, gid):
        # Ownership cannot be faked with real files without root.
        st = mock.MagicMock()
        st.st_uid = uid
        st.st_gid = gid
        return mock.patch("virtme.commands.run.os.stat", return_value=st)

    def test_returns_owner(self):
        for owner in ((1000, 1000), (0, 0), (1001, 1000), (1000, 1001), (1001, 1001)):
            with self.subTest(owner=owner), self._stat_as(*owner):
                self.assertEqual(rootfs_ownership(self.tempdir.name), owner)

    def test_mixed_ownership_rootfs_returns_none(self):
        # The top-level directory belongs to the user, but not its content.
        os.mkdir(os.path.join(self.tempdir.name, "usr"))
        foreign = mock.MagicMock(st_uid=0, st_gid=0, st_mode=stat.S_IFDIR)
        with (
            self._stat_as(1000, 1000),
            mock.patch("virtme.commands.run.os.lstat", return_value=foreign),
        ):
            self.assertIsNone(rootfs_ownership(self.tempdir.name))

    def test_uniformly_owned_rootfs_returns_owner(self):
        os.mkdir(os.path.join(self.tempdir.name, "usr"))
        owned = mock.MagicMock(st_uid=1000, st_gid=1000, st_mode=stat.S_IFDIR)
        with (
            self._stat_as(1000, 1000),
            mock.patch("virtme.commands.run.os.lstat", return_value=owned),
        ):
            self.assertEqual(rootfs_ownership(self.tempdir.name), (1000, 1000))

    def test_root_owned_symlink_does_not_defeat_detection(self):
        # A `chown -R` without `-h` leaves the merged-usr symlinks (bin,
        # sbin, lib) behind as root-owned; their ownership must not
        # disable the mapping of a user-owned rootfs.
        root_owned_symlink = mock.MagicMock(st_uid=0, st_gid=0, st_mode=stat.S_IFLNK)
        user_owned_dir = mock.MagicMock(st_uid=1000, st_gid=1000, st_mode=stat.S_IFDIR)

        def fake_lstat(path):
            if os.path.basename(path) in ("bin", "sbin", "lib"):
                return root_owned_symlink
            return user_owned_dir

        with (
            self._stat_as(1000, 1000),
            mock.patch("virtme.commands.run.os.lstat", side_effect=fake_lstat),
        ):
            self.assertEqual(rootfs_ownership(self.tempdir.name), (1000, 1000))

    def test_missing_path_returns_none(self):
        path = os.path.join(self.tempdir.name, "nonexistent")
        self.assertIsNone(rootfs_ownership(path))


class GuestRootTranslationTests(unittest.TestCase):
    """guest_root_translation() resolves the guest root's host identity."""

    def setUp(self):
        super().setUp()
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)

    @staticmethod
    def _stat_as(uid, gid):
        # Ownership cannot be faked with real files without root.
        st = mock.MagicMock()
        st.st_uid = uid
        st.st_gid = gid
        return mock.patch("virtme.commands.run.os.stat", return_value=st)

    @contextlib.contextmanager
    def _running_as(self, uid, gid, groups=(1000,)):
        # os is a shared module: this is also the identity that
        # may_translate_owner() sees.
        with (
            mock.patch("virtme.commands.run.os.getuid", return_value=uid),
            mock.patch("virtme.commands.run.os.getgid", return_value=gid),
            mock.patch("virtme.commands.run.os.getgroups", return_value=list(groups)),
        ):
            yield

    def test_user_owned_rootfs_is_mapped_to_its_owner(self):
        # The tempdir has none of the well-known top-level entries, so only
        # its own (mocked) ownership counts. Root may translate to any
        # owner, an unprivileged user only to themselves.
        for identity, owner in (((0, 0), (1234, 5678)), ((1000, 1000), (1000, 1000))):
            with self.subTest(identity=identity, owner=owner):
                with self._stat_as(*owner), self._running_as(*identity):
                    self.assertEqual(
                        guest_root_translation(
                            self.tempdir.name, "the owner of the rootfs"
                        ),
                        (*owner, "the owner of the rootfs"),
                    )

    def test_root_owned_rootfs_is_mapped_to_the_invoking_user(self):
        # An unprivileged daemon cannot act as root: the invoking user is
        # the only identity the guest root can have on the host.
        with self._stat_as(0, 0), self._running_as(1000, 1000):
            self.assertEqual(
                guest_root_translation(self.tempdir.name, "the owner of the rootfs"),
                (1000, 1000, "the invoking user"),
            )

    def test_root_owned_rootfs_is_not_mapped_for_root(self):
        with self._stat_as(0, 0), self._running_as(0, 0):
            self.assertIsNone(
                guest_root_translation(self.tempdir.name, "the owner of the rootfs")
            )

    def test_undeterminable_owner_is_mapped_to_the_invoking_user(self):
        path = os.path.join(self.tempdir.name, "nonexistent")
        with self._running_as(1000, 1000):
            self.assertEqual(
                guest_root_translation(path, "the owner of the rootfs"),
                (1000, 1000, "the invoking user"),
            )

    def test_mixed_ownership_is_mapped_to_the_invoking_user(self):
        os.mkdir(os.path.join(self.tempdir.name, "usr"))
        foreign = mock.MagicMock(st_uid=0, st_gid=0, st_mode=stat.S_IFDIR)
        with (
            self._stat_as(1000, 1000),
            mock.patch("virtme.commands.run.os.lstat", return_value=foreign),
            self._running_as(1000, 1000),
        ):
            self.assertEqual(
                guest_root_translation(
                    self.tempdir.name, "the owner of the shared directory"
                ),
                (1000, 1000, "the invoking user"),
            )

    def test_untranslatable_owner_is_not_mapped(self):
        stderr = io.StringIO()
        with (
            self._stat_as(1001, 1001),
            self._running_as(1000, 1000),
            contextlib.redirect_stderr(stderr),
        ):
            self.assertIsNone(
                guest_root_translation(self.tempdir.name, "the owner of the rootfs")
            )
        self.assertIn("is owned by uid 1001, gid 1001", stderr.getvalue())
        self.assertIn("not mapped to the owner of the rootfs", stderr.getvalue())


class TranslateOwnerPermissionTests(unittest.TestCase):
    # The identity of the invoking user cannot be faked without root.

    def test_root_may_translate_any_owner(self):
        with mock.patch("virtme_ng.virtiofs.os.getuid", return_value=0):
            for owner in ((1000, 1000), (1234, 5678), (0, 1000), (1000, 0), (0, 0)):
                with self.subTest(owner=owner):
                    self.assertTrue(VirtioFS.may_translate_owner(owner))

    def test_unprivileged_user_may_translate_own_and_zero_components(self):
        with (
            mock.patch("virtme_ng.virtiofs.os.getuid", return_value=1000),
            mock.patch("virtme_ng.virtiofs.os.getgid", return_value=1000),
            mock.patch("virtme_ng.virtiofs.os.getgroups", return_value=[1000]),
        ):
            for owner in ((1000, 1000), (1000, 0), (0, 1000), (0, 0)):
                with self.subTest(owner=owner):
                    self.assertTrue(VirtioFS.may_translate_owner(owner))

    def test_unprivileged_user_may_translate_supplementary_group(self):
        with (
            mock.patch("virtme_ng.virtiofs.os.getuid", return_value=1000),
            mock.patch("virtme_ng.virtiofs.os.getgid", return_value=1000),
            mock.patch("virtme_ng.virtiofs.os.getgroups", return_value=[1000, 500]),
        ):
            self.assertTrue(VirtioFS.may_translate_owner((1000, 500)))

    def test_unprivileged_user_may_not_translate_foreign_owner(self):
        with (
            mock.patch("virtme_ng.virtiofs.os.getuid", return_value=1000),
            mock.patch("virtme_ng.virtiofs.os.getgid", return_value=1000),
            mock.patch("virtme_ng.virtiofs.os.getgroups", return_value=[1000]),
        ):
            for owner in ((1001, 1000), (1000, 1001), (1001, 1001), (1234, 5678)):
                with self.subTest(owner=owner):
                    self.assertFalse(VirtioFS.may_translate_owner(owner))


class CleanupTests(FakeDaemonTestCase):
    def test_stops_the_daemon_and_removes_its_temp_files(self):
        fs, started = self.start_daemon(_rootfs_config())
        self.assertTrue(started)

        fs.shutdown()

        # The daemon was told to stop and is gone, together with its files.
        fs.proc.wait(timeout=5)
        self.assertLess(fs.proc.returncode, 0)
        self.assertFalse(os.path.exists(fs.sock))
        self.assertFalse(os.path.exists(fs.sock + ".pid"))

    def test_kills_a_daemon_that_ignores_sigterm(self):
        self.make_daemon(self.daemon, role="stubborn")
        fs, started = self.start_daemon(_rootfs_config())
        self.assertTrue(started)

        fs.shutdown()

        # SIGTERM was ignored: the daemon was killed anyway.
        fs.proc.wait(timeout=5)
        self.assertLess(fs.proc.returncode, 0)
        self.assertFalse(os.path.exists(fs.sock))


class NoRootUidMapCliTests(unittest.TestCase):
    def test_parser_defaults_to_false(self):
        parser = make_parser()
        args = parser.parse_args(["--mods", "none"])
        self.assertFalse(args.no_root_uid_map)

    def test_parser_accepts_flag(self):
        parser = make_parser()
        args = parser.parse_args(["--mods", "none", "--no-root-uid-map"])
        self.assertTrue(args.no_root_uid_map)


if __name__ == "__main__":
    unittest.main()
