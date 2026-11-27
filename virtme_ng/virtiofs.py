"""virtme-ng: virtiofsd daemon management and virtio-fs QEMU export arguments."""

import atexit
import ctypes
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
from functools import lru_cache
from shutil import which
from time import monotonic, sleep

from virtme import architectures

_VIRTIOFSD_PATH: dict[tuple[str, str | None], str | None] = {}

_PR_SET_PDEATHSIG = 1


def _die_with_parent():
    """Return a Popen(preexec_fn=) that SIGTERMs the child when its parent dies.

    This also covers a SIGKILLed parent, which runs no exit handler. The
    kernel ties the signal to the spawning *thread*, so that must be a
    long-lived one (the main thread).
    """
    # Load libc here: the returned function runs between fork and exec, where
    # nothing but already prepared code should be executed.
    prctl = ctypes.CDLL(None).prctl
    parent_pid = os.getpid()

    def preexec():
        # Without this, the daemon would survive a SIGKILLed virtme-run.
        # prctl() returns -1 on error.
        if prctl(_PR_SET_PDEATHSIG, signal.SIGTERM) != 0:
            os._exit(1)
        # The parent may have died before the prctl() call took effect.
        if os.getppid() != parent_pid:
            os._exit(1)

    return preexec


class VirtioFSConfig:
    def __init__(
        self,
        path: str,
        mount_tag: str,
        *,
        rw: bool = False,
        posix_acl: bool = False,
        translate_uid: int | None = None,
        translate_gid: int | None = None,
        translate_desc: str | None = None,
    ):
        self.path = path
        self.mount_tag = mount_tag
        self.rw = rw
        self.posix_acl = posix_acl
        # 0 would be the identity (the guest root already has root's
        # uid/gid): normalize it away.
        self.translate_uid = translate_uid if translate_uid != 0 else None
        self.translate_gid = translate_gid if translate_gid != 0 else None
        # What the guest root is mapped to, for the messages (e.g. "the
        # owner of the rootfs"); the ids are always part of them.
        self.translate_desc = translate_desc

    @property
    def readonly(self) -> bool:
        return not self.rw

    @property
    def translate_target(self) -> str | None:
        """The guest root's mapping target as used in the messages.

        None when the guest root is not mapped at all.
        """
        parts = []
        if self.translate_uid is not None:
            parts.append(f"uid {self.translate_uid}")
        if self.translate_gid is not None:
            parts.append(f"gid {self.translate_gid}")
        if not parts:
            return None
        target = ", ".join(parts)
        if self.translate_desc:
            target = f"{self.translate_desc} ({target})"
        return target


class VirtioFSState:
    def __init__(
        self,
        guest_tools_path,
        memory=None,
        memory_configured=False,
        virtiofsd_path=None,
    ):
        self.guest_tools_path = guest_tools_path
        self.memory = memory
        self.memory_configured = memory_configured
        self.virtiofsd_path = virtiofsd_path


class VirtioFS:
    def __init__(self, guest_tools_path, virtiofsd_path=None):
        self.sock = None
        self.proc = None
        self.guest_tools_path = guest_tools_path
        self.virtiofsd_path = virtiofsd_path

    def shutdown(self):
        """Stop the daemon (SIGTERM, then SIGKILL) and remove its temp files."""
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Even SIGKILL may fail; shutdown must never abort the VM
                # launch/exit path.
                try:
                    self.proc.kill()
                    self.proc.wait(timeout=5)
                except (subprocess.TimeoutExpired, PermissionError):
                    pass
            except PermissionError:
                pass
        # Clean up temp files (virtiofsd creates a pid file next to the socket
        # in --socket-path mode; remove it as well, even though it is unused)
        if self.sock is not None:
            for file_path in (self.sock, f"{self.sock}.pid"):
                try:
                    os.remove(file_path)
                except OSError:
                    pass

    def is_virtiofsd_available(self) -> bool:
        """Whether a usable virtiofsd binary is installed.

        Unlike start(), this does not attempt to launch the daemon.
        """
        return self._get_virtiofsd_path() is not None

    def _get_virtiofsd_path(self):
        cache_key = (self.guest_tools_path, self.virtiofsd_path)
        if cache_key in _VIRTIOFSD_PATH:
            return _VIRTIOFSD_PATH[cache_key]

        # Define the possible virtiofsd paths.
        #
        # NOTE: do not use the C implementation of qemu's virtiofsd, because it
        # doesn't support unprivileged-mode execution and it would be totally
        # unsafe to export the whole rootfs of the host running as root.
        #
        # Instead, always rely on the Rust implementation of virtio-fs:
        # https://gitlab.com/virtio-fs/virtiofsd
        #
        # This project is receiving the most attention for new feature development
        # and the daemon is able to export the entire root filesystem of the host
        # as non-privileged user.
        #
        # Starting with version 8.0, qemu will not ship the C implementation of
        # virtiofsd anymore, allowing to use the Rust daemon installed in the the
        # same path (/usr/lib/qemu/virtiofsd), so also consider this one in the
        # list of possible paths.
        #
        # We can detect if the qemu implementation is installed in /usr/lib/qemu,
        # simply by running the command with --version as non-root. If it returns
        # an error it means that we are using the qemu daemon and we just skip it.
        #
        # An explicitly requested daemon is the only candidate: silently
        # using a different one would be surprising.
        if self.virtiofsd_path is not None:
            possible_paths = (self.virtiofsd_path,)
        else:
            possible_paths = (
                f"{self.guest_tools_path}/bin/virtiofsd",
                which("virtiofsd"),
                "/usr/libexec/virtiofsd",
                "/usr/lib/virtiofsd/virtiofsd",
                "/usr/lib/virtiofsd",
                "/usr/lib/qemu/virtiofsd",
            )
        for path in possible_paths:
            if path and os.path.exists(path) and os.access(path, os.X_OK):
                try:
                    subprocess.check_call(
                        [path, "--version"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    _VIRTIOFSD_PATH[cache_key] = path
                    return path
                except (subprocess.CalledProcessError, OSError):
                    pass
        if self.virtiofsd_path is not None:
            sys.stderr.write(
                f"virtme-run: '{self.virtiofsd_path}' is not a usable virtiofsd\n"
            )
        _VIRTIOFSD_PATH[cache_key] = None
        return None

    @staticmethod
    def _is_socket(path):
        """Whether PATH is a socket owned by the current user.

        The path is visible in /proc/*/cmdline: a local user could race the
        daemon and bind their own socket there, which QEMU would connect to.
        """
        try:
            st = os.stat(path)
            return stat.S_ISSOCK(st.st_mode) and st.st_uid == os.geteuid()
        except OSError:
            return False

    @staticmethod
    @lru_cache
    def _get_virtiofsd_help(path) -> str:
        """Cached `--help` output of the virtiofsd at PATH ("" if unavailable)."""
        try:
            result = subprocess.run(
                [path, "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        except OSError:
            return ""
        return result.stdout if result.returncode == 0 else ""

    @staticmethod
    def _supports_flag(path, flag) -> bool:
        """Whether the `--help` output lists FLAG as a whole word.

        The formatting differs across virtiofsd versions, and --readonly
        must not match --readonly-mode.
        """
        return (
            re.search(
                rf"{re.escape(flag)}(?![\w-])", VirtioFS._get_virtiofsd_help(path)
            )
            is not None
        )

    @staticmethod
    def _supports_readonly(path):
        return VirtioFS._supports_flag(path, "--readonly")

    @staticmethod
    def may_translate_owner(owner: tuple[int, int]) -> bool:
        """Whether the current user may translate the guest root to OWNER.

        An unprivileged virtiofsd can only act as the invoking user: the uid
        must be theirs, the gid one of their groups (0 is not translated).
        Root may translate to anyone.
        """
        if os.getuid() == 0:
            return True
        owner_uid, owner_gid = owner
        if owner_uid not in (0, os.getuid()):
            return False
        return owner_gid in (0, os.getgid(), *os.getgroups())

    def _report_9p_fallback(self, verbose, target=None):
        # A lost UID mapping changes what the guest sees: always announce it.
        if verbose or target:
            sys.stderr.write("virtme-run: failed to start virtiofsd, fallback to 9p\n")
        if target:
            sys.stderr.write(f"virtme-run: the guest root is not mapped to {target}\n")

    def start(self, config: VirtioFSConfig, verbose=True):
        translate_opt = []
        # What the guest root is mapped to, as used in the messages.
        target = config.translate_target

        virtiofsd_path = self._get_virtiofsd_path()
        if virtiofsd_path is None:
            self._report_9p_fallback(verbose, target)
            return False

        # virtiofsd located, try to start the daemon as non-privileged user.
        sock_fd, self.sock = tempfile.mkstemp(prefix="virtme", suffix=".sock")
        os.close(sock_fd)

        self.shutdown()
        atexit.register(self.shutdown)

        # Export the whole root fs of the host, do not enable sandbox, otherwise we
        # would get permission errors.
        if verbose:
            stdout = stderr = None
        else:
            stdout = stderr = subprocess.DEVNULL
        # Only enable --posix-acl when using an external root filesystem.
        # When sharing the host root (/), --posix-acl breaks UID/GID translation
        # which causes authentication failures (e.g., unix_chkpwd cannot obtain
        # user info).
        acl_opt = ["--posix-acl"] if config.posix_acl else []
        if target is not None:
            wanted = []
            if config.translate_uid is not None:
                wanted.append(("--translate-uid", config.translate_uid))
            if config.translate_gid is not None:
                wanted.append(("--translate-gid", config.translate_gid))

            if not all(self._supports_flag(virtiofsd_path, flag) for flag, _ in wanted):
                missing = "/".join(
                    flag
                    for flag, _ in wanted
                    if not self._supports_flag(virtiofsd_path, flag)
                )
                sys.stderr.write(
                    f"virtme-run: virtiofsd does not support {missing}; "
                    f"the guest root is not mapped to {target}\n"
                )
            else:
                # UID/GID translation cannot be combined with POSIX ACLs:
                # drop them in this case.
                if acl_opt:
                    sys.stderr.write(
                        "virtme-run: disabling POSIX ACLs for UID/GID translation\n"
                    )
                    acl_opt = []
                translate_opt = [f"{flag}=map:0:{host}:1" for flag, host in wanted]
                sys.stderr.write(f"virtme-run: mapping guest root to {target}\n")
        readonly_opt = (
            ["--readonly"]
            if config.readonly and self._supports_readonly(virtiofsd_path)
            else []
        )
        virtiofsd_cmd = [
            virtiofsd_path,
            "--syslog",
            "--no-announce-submounts",
            *acl_opt,
            *translate_opt,
            *readonly_opt,
            f"--socket-path={self.sock}",
            f"--shared-dir={config.path}",
            "--sandbox=none",
            f"--cache={'auto' if config.rw else 'always'}",
        ]
        try:
            # Keep the daemon off the terminal's stdin and in its own process
            # group, but in the same session: a Ctrl-C meant for QEMU must
            # not kill it before the guest shuts down cleanly, while it must
            # still die with virtme-run. preexec_fn is safe here: no other
            # thread runs yet, and _die_with_parent() only calls prctl() and
            # getppid() in the child.
            self.proc = subprocess.Popen(  # pylint: disable=W1509
                virtiofsd_cmd,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                process_group=0,
                preexec_fn=_die_with_parent(),
            )
        except OSError:
            # The caller expects a graceful fallback to 9p, not an exception.
            self._report_9p_fallback(verbose, target if translate_opt else None)
            return False
        deadline = monotonic() + 3.1
        waiting_logged = False
        while not self._is_socket(self.sock):
            remaining = deadline - monotonic()
            if remaining <= 0 or self.proc.poll() is not None:
                self._report_9p_fallback(verbose, target if translate_opt else None)
                return False
            if verbose and not waiting_logged:
                sys.stderr.write("virtme: waiting for virtiofsd to start\n")
                waiting_logged = True
            sleep(min(0.01, remaining))
        # The socket path shows up at bind() time: a daemon that exits right
        # after creating it must not be mistaken for a ready one.
        if self.proc.poll() is not None:
            self._report_9p_fallback(verbose, target if translate_opt else None)
            return False
        return True


def ensure_virtiofs_memory(
    arch: architectures.Arch,
    qemuargs: list[str],
    state: VirtioFSState,
) -> None:
    if state.memory_configured:
        return

    memory = state.memory if state.memory is not None else "128M"
    qemuargs.extend(["-object", f"memory-backend-memfd,id=mem,size={memory},share=on"])
    if arch.numa_support():
        qemuargs.extend(["-numa", "node,memdev=mem"])
    else:
        qemuargs.extend(["-machine", "memory-backend=mem"])
    state.memory_configured = True


def export_virtiofs(
    arch: architectures.Arch,
    qemuargs: list[str],
    state: VirtioFSState,
    config: VirtioFSConfig,
    verbose=False,
) -> bool:
    if not arch.virtiofs_support():
        # start() is never reached without virtiofs support, so a lost
        # mapping has to be announced here.
        if config.translate_target is not None:
            sys.stderr.write(
                "virtme-run: no virtiofs support on this architecture: "
                f"the guest root is not mapped to {config.translate_target}\n"
            )
        return False

    # Try to start virtiofsd daemon
    virtio_fs = VirtioFS(state.guest_tools_path, state.virtiofsd_path)
    ret = virtio_fs.start(config, verbose=verbose)
    if not ret:
        return False

    # Adjust qemu options to use virtiofsd
    fsid = f"virtfs{len(qemuargs)}"
    vhost_dev_type = arch.vhost_dev_type("user-fs")

    qemuargs.extend(["-chardev", f"socket,id=char{fsid},path={virtio_fs.sock}"])
    qemuargs.extend(
        ["-device", f"{vhost_dev_type},chardev=char{fsid},tag={config.mount_tag}"]
    )

    ensure_virtiofs_memory(arch, qemuargs, state)
    return True
