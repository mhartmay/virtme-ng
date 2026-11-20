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

from virtme.commands.run import rootfs_ownership
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

    def start_daemon(self, config, verbose=False):
        """start() the daemon for CONFIG; it is cleaned up afterwards."""
        fs = VirtioFS(self.guest_tools)

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
                _rootfs_config(posix_acl=True, translate_uid=1234, translate_gid=5678)
            )
        self.assertTrue(started)
        (argv,) = self.daemon_runs(self.daemon)
        self.assertIn("--translate-uid=map:0:1234:1", argv)
        self.assertIn("--translate-gid=map:0:5678:1", argv)
        # UID/GID translation cannot be combined with POSIX ACLs.
        self.assertNotIn("--posix-acl", argv)
        self.assertIn("mapping guest root", stderr.getvalue())
        self.assertIn("disabling POSIX ACLs", stderr.getvalue())

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
        self.assertIn("not mapped to the rootfs owner", stderr.getvalue())

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
        self.assertIn("not mapped to the rootfs owner", stderr.getvalue())

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
        self.assertIn("not mapped to the rootfs owner", stderr.getvalue())

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


if __name__ == "__main__":
    unittest.main()
