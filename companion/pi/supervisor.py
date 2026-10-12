# Codex, gpt-6-astra, Jetson seat, 2026-10-11 23:32:40 EDT
"""Independent monotonic watchdog and isolated, killable driver process."""
import ctypes
import math
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
import traceback

from .board import FakeBoard


def _send_quietly(connection, message):
    try:
        connection.send(message)
    except Exception:
        pass


def _worker(connection, factory, parent_pid):
    # Linux PR_SET_PDEATHSIG: parent crash/SIGKILL must also close the child FD.
    # spawn avoids inheriting the TCP listener, client, or journal lock.
    # Deployment is Linux only; prctl does not exist on macOS, skipped there so tests reach the factory.
    if sys.platform.startswith('linux'):
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), 'parent-death signal unavailable')
    ppid = os.getppid()
    if ppid != parent_pid:
        print('driver worker pid %d: parent mismatch ppid=%d expected=%d'
              % (os.getpid(), ppid, parent_pid), file=sys.stderr, flush=True)
        _send_quietly(connection, {'ready': False, 'reason': 'parent mismatch',
                                   'ppid': ppid, 'expected': parent_pid})
        connection.close()
        return
    sys.stdout = sys.stderr
    board = None
    try:
        try:
            # Exception text travels to the parent and journal: factories must not put secret values in exception messages.
            board = factory()
        except BaseException as exc:
            trace = traceback.format_exc()
            print('driver worker pid %d: factory failed\n%s' % (os.getpid(), trace),
                  file=sys.stderr, flush=True)
            _send_quietly(connection, {'ready': False, 'reason': 'factory failed',
                                       'exc_type': type(exc).__name__,
                                       'exc': str(exc)[:500], 'trace': trace[-2000:]})
            if isinstance(exc, (SystemExit, KeyboardInterrupt)):
                raise
            raise SystemExit(1)
        connection.send({'ready': True, 'kind': getattr(board, 'adapter_kind', 'undeclared')})
        try:
            while True:
                operation, args = connection.recv()
                if operation == 'ping':
                    result = board.health() if hasattr(board, 'health') else {'alive': True}
                elif operation == 'status':
                    result = board.status() if hasattr(board, 'status') else {}
                elif operation == 'probe':
                    result = board.probe(args) if hasattr(board, 'probe') else {'ok': False, 'reason': 'probe_unavailable'}
                elif operation == 'action':
                    result = board.execute(args['command'], args['args'])
                elif operation == 'snapshot':
                    result = board.snapshot()
                else:
                    raise ValueError('unknown driver operation')
                connection.send(result)
        except EOFError:
            pass
    finally:
        if board is not None:
            board.close()
        connection.close()


class Supervisor:
    """One driver per link. Faults latch until a new disarmed session is made.

    A thread kills the driver even if the TCP reader, response writer or journal
    stalls. A separate response timeout catches a frozen/blocked driver.
    """
    def __init__(self, factory=FakeBoard, lease_seconds=2.0, response_timeout=0.5):
        for value in (lease_seconds, response_timeout):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0.05 <= value <= 5:
                raise ValueError('timeouts must be finite, 0.05..5 seconds')
        self.lease_seconds, self.response_timeout = lease_seconds, response_timeout
        self._deadline = time.monotonic() + lease_seconds
        self._state_lock = threading.Lock()
        self._call_lock = threading.Lock()
        self._done = threading.Event()
        self.reason = None
        ctx = mp.get_context('spawn')
        self.connection, child = ctx.Pipe()
        self.process = ctx.Process(target=_worker, args=(child, factory, os.getpid()))
        self.process.start()
        child.close()
        self._watch = threading.Thread(target=self._watchdog, daemon=True)
        self._watch.start()
        try:
            # Import/startup has a separate bounded budget, never arms hardware.
            if not self.connection.poll(min(2.0, lease_seconds)):
                raise TimeoutError('driver startup timeout')
            try:
                ready = self.connection.recv()
            except EOFError:
                raise ConnectionError('driver child closed the pipe before ready') from None
            if isinstance(ready, dict) and ready.get('ready') is False:
                detail = ', '.join('%s=%s' % (k, ready[k]) for k in ('reason', 'exc_type', 'exc', 'ppid', 'expected')
                                   if k in ready)
                raise ConnectionError('driver startup failed in child: ' + detail)
            if not isinstance(ready, dict) or ready.get('ready') is not True:
                raise ConnectionError('invalid driver startup')
            self.kind = ready.get('kind', 'undeclared')
        except BaseException as exc:
            # Join briefly so the journal names the child's exit code (-9 = SIGKILL).
            self.process.join(timeout=0.5)
            exitcode, watchdog_reason = self.process.exitcode, self.reason
            print('driver startup failed: %s: %s, child exitcode=%s, watchdog_reason=%s'
                  % (type(exc).__name__, exc, exitcode, watchdog_reason), file=sys.stderr, flush=True)
            self.close('driver startup failed')
            if isinstance(exc, (ConnectionError, TimeoutError)):
                raise type(exc)('%s; child exitcode=%s, watchdog_reason=%s'
                                % (exc, exitcode, watchdog_reason)) from exc
            raise

    @property
    def healthy(self):
        return self.reason is None and self.process.is_alive()

    def renew(self):
        with self._state_lock:
            if self.reason is not None or time.monotonic() >= self._deadline:
                raise ConnectionError(self.reason or 'heartbeat lease already expired')
            self._deadline = time.monotonic() + self.lease_seconds

    def _watchdog(self):
        while not self._done.wait(0.01):
            with self._state_lock:
                expired = time.monotonic() >= self._deadline
            if expired or not self.process.is_alive():
                self.trip('Jetson heartbeat missed' if expired else 'driver exited')
                return

    def trip(self, reason):
        with self._state_lock:
            if self.reason is None:
                self.reason = reason
            self._done.set()
            if self.process.is_alive():
                self.process.kill()

    def call(self, operation, args=None):
        with self._call_lock:
            if not self.healthy:
                raise ConnectionError(self.reason or 'driver unavailable')
            try:
                self.connection.send((operation, args or {}))
                if not self.connection.poll(self.response_timeout):
                    raise TimeoutError('driver response missed')
                result = self.connection.recv()
                if not self.healthy:
                    raise ConnectionError(self.reason or 'driver exited')
                return result
            except (OSError, EOFError, TimeoutError) as exc:
                self.trip('driver response failed')
                raise ConnectionError('driver response failed; outcome unknown') from exc

    def close(self, reason='link closed'):
        self.trip(reason)
        self.process.join(timeout=1)
        if threading.current_thread() is not self._watch:
            self._watch.join(timeout=1)
        self.connection.close()
