"""ThreadPoolExecutor with daemon workers so the process can exit while I/O tasks wind down."""

from __future__ import annotations

import threading
import weakref
from concurrent.futures.thread import ThreadPoolExecutor, _threads_queues, _worker


class DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """
    Same as ThreadPoolExecutor, but worker threads are daemon=True.

    Non-daemon pool threads otherwise keep the interpreter alive after the main thread
    finishes UI teardown (e.g. Ctrl+C during microphone capture or playback).
    """

    def _adjust_thread_count(self) -> None:
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = "%s_%d" % (self._thread_name_prefix or self, num_threads)
            if getattr(self, "_create_worker_context", None) is not None:
                worker_args = (
                    weakref.ref(self, weakref_cb),
                    self._create_worker_context(),
                    self._work_queue,
                )
            else:
                worker_args = (
                    weakref.ref(self, weakref_cb),
                    self._work_queue,
                    self._initializer,
                    self._initargs,
                )
            t = threading.Thread(
                name=thread_name,
                target=_worker,
                args=worker_args,
                daemon=True,
            )
            t.start()
            self._threads.add(t)
            _threads_queues[t] = self._work_queue
