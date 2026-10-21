"""Tests for a clean ONVIF controller shutdown.

Each ONVIFCamera holds aiohttp sessions, and the controller runs a config poll
that never returns. Left open or pending, they are garbage collected during
shutdown and logged as "Unclosed client session" or "Task was destroyed but it
is pending!" after the log queue has already gone.
"""

import asyncio
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp

from frigate.camera import PTZMetrics
from frigate.ptz.onvif import OnvifController


def _drain(controller: OnvifController) -> None:
    """Cancel what a test left on the loop, so it is not logged at exit."""
    loop = controller.loop

    if controller.loop_thread.is_alive():
        loop.call_soon_threadsafe(loop.stop)
        controller.loop_thread.join(timeout=5)

    tasks = asyncio.all_tasks(loop)

    for task in tasks:
        task.cancel()

    # an empty gather would bind to another loop
    if tasks:
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))

    loop.close()


def _make_controller(*camera_names: str) -> OnvifController:
    """Build a controller with a running loop thread and mocked ONVIF clients."""
    controller = OnvifController.__new__(OnvifController)
    controller.loop = asyncio.new_event_loop()
    controller.loop_thread = threading.Thread(
        target=controller.loop.run_forever, daemon=True
    )
    controller.loop_thread.start()
    controller.config_subscriber = MagicMock()
    controller.cams = {
        name: {"onvif": MagicMock(close=AsyncMock())} for name in camera_names
    }
    return controller


class TestOnvifControllerClose(unittest.TestCase):
    def _controller(self, *camera_names: str) -> OnvifController:
        controller = _make_controller(*camera_names)
        self.addCleanup(_drain, controller)
        return controller

    def test_closes_every_camera_session(self) -> None:
        controller = self._controller("front", "back")

        controller.close()

        for cam in controller.cams.values():
            cam["onvif"].close.assert_awaited_once()
        self.assertFalse(controller.loop_thread.is_alive())

    def test_failing_close_does_not_skip_the_others(self) -> None:
        controller = self._controller("front", "back")
        controller.cams["front"]["onvif"].close.side_effect = RuntimeError("boom")

        controller.close()

        controller.cams["back"]["onvif"].close.assert_awaited_once()
        self.assertFalse(controller.loop_thread.is_alive())

    def test_stops_the_config_poll(self) -> None:
        # the poll never returns, and a task destroyed while still pending is
        # logged during shutdown too
        controller = self._controller("front")

        async def start_poll() -> asyncio.Task:
            return asyncio.create_task(asyncio.sleep(3600))

        controller.config_poll = asyncio.run_coroutine_threadsafe(
            start_poll(), controller.loop
        ).result(timeout=5)

        controller.close()

        self.assertTrue(controller.config_poll.cancelled())
        self.assertFalse(controller.loop_thread.is_alive())

    def test_leaves_other_tasks_on_the_loop_alone(self) -> None:
        # the autotracker runs its own tasks on this loop, and its threads wait on
        # them without handling a cancellation
        controller = self._controller("front")
        other = asyncio.run_coroutine_threadsafe(asyncio.sleep(3600), controller.loop)

        controller.close()

        self.assertFalse(other.cancelled())

    def test_camera_whose_client_failed_is_skipped(self) -> None:
        # a failed ONVIFCamera constructor leaves the camera in failed_cams only
        controller = self._controller("front")
        controller.failed_cams = {"back": {"retry_attempts": 0}}

        controller.close()

        controller.cams["front"]["onvif"].close.assert_awaited_once()
        self.assertFalse(controller.loop_thread.is_alive())

    def test_hanging_session_close_does_not_hang_shutdown(self) -> None:
        controller = self._controller("front")
        controller.shutdown_timeout = 0.2

        async def never_returns() -> None:
            await asyncio.sleep(3600)

        controller.cams["front"]["onvif"].close = never_returns
        closer = threading.Thread(target=controller.close, daemon=True)

        closer.start()
        closer.join(timeout=10)

        self.assertFalse(closer.is_alive(), "close() hung")
        self.assertFalse(controller.loop_thread.is_alive())

    def test_stops_the_poll_that_init_started(self) -> None:
        # the poll that the real __init__ starts, not one a test injects
        async def poll_forever(self) -> None:
            await asyncio.sleep(3600)

        with (
            patch("frigate.ptz.onvif.CameraConfigUpdateSubscriber"),
            patch.object(OnvifController, "_poll_config_updates", poll_forever),
        ):
            controller = OnvifController(MagicMock(cameras={}), {})

        self.addCleanup(_drain, controller)
        # one round trip, so the loop has run the callbacks __init__ queued
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0), controller.loop).result(
            timeout=5
        )
        poll = controller.config_poll
        self.assertIsNotNone(poll)

        controller.close()

        self.assertTrue(poll.cancelled())


class TestOnvifControllerCloseWhileMoving(unittest.TestCase):
    """A PTZ is moving when close() runs, so the autotracker waits for its motor."""

    def setUp(self) -> None:
        self.controller = _make_controller()
        self.addCleanup(_drain, self.controller)
        self.session_closed = False

        async def get_status(request):
            # a closed aiohttp session raises before awaiting anything
            if self.session_closed:
                raise RuntimeError("Session is closed")

            await asyncio.sleep(0.01)
            raise ConnectionError("still moving, status unknown")

        async def close_session():
            # the real ONVIFCamera.close() yields while it closes each session
            self.session_closed = True
            await asyncio.sleep(0.05)

        self.controller.cams = {
            "ptz": {
                "onvif": MagicMock(close=AsyncMock(side_effect=close_session)),
                "init": True,
                "ptz": SimpleNamespace(GetStatus=get_status),
                "status_request": MagicMock(),
            }
        }
        self.controller.ptz_metrics = {"ptz": PTZMetrics(autotracker_enabled=True)}
        self.controller.ptz_metrics["ptz"].motor_stopped.clear()
        self.controller.status_locks = {}
        asyncio.run_coroutine_threadsafe(
            self._create_lock(), self.controller.loop
        ).result(timeout=5)

    async def _create_lock(self) -> None:
        self.controller.status_locks["ptz"] = asyncio.Lock()

    async def _wait_for_motor(self) -> None:
        # the autotracker's wait after every move and before a preset return
        while not self.controller.ptz_metrics["ptz"].motor_stopped.is_set():
            await self.controller.get_camera_status("ptz")

    def test_close_returns_while_the_autotracker_waits_for_the_motor(self) -> None:
        wait = asyncio.run_coroutine_threadsafe(
            self._wait_for_motor(), self.controller.loop
        )
        time.sleep(0.1)
        closer = threading.Thread(target=self.controller.close, daemon=True)

        closer.start()
        closer.join(timeout=10)

        self.assertFalse(closer.is_alive(), "close() hung")
        self.assertTrue(wait.done())
        self.assertTrue(self.session_closed)

    def test_status_after_close_releases_waiters(self) -> None:
        self.controller.stopping = True

        asyncio.run_coroutine_threadsafe(
            self.controller.get_camera_status("ptz"), self.controller.loop
        ).result(timeout=5)

        self.assertTrue(self.controller.ptz_metrics["ptz"].motor_stopped.is_set())


class TestOnvifControllerQuietWhileStopping(unittest.TestCase):
    """Requests that close() aborts must not be reported as camera faults."""

    def test_aborted_connection_is_not_reported(self) -> None:
        controller = OnvifController.__new__(OnvifController)
        onvif = MagicMock(
            update_xaddrs=AsyncMock(side_effect=aiohttp.ServerDisconnectedError())
        )
        controller.cams = {"ptz": {"onvif": onvif}}
        controller.stopping = True

        with self.assertNoLogs("frigate.ptz.onvif", level="WARNING"):
            self.assertFalse(asyncio.run(controller._init_onvif("ptz")))

    def test_aborted_probe_is_not_reported(self) -> None:
        controller = OnvifController.__new__(OnvifController)
        controller.config = MagicMock()
        controller.cams = {"ptz": {"init": False}}
        controller.failed_cams = {}
        controller.max_retries = 5
        controller.reset_timeout = 900
        controller._init_onvif = AsyncMock(return_value=False)
        controller.stopping = True

        with self.assertNoLogs("frigate.ptz.onvif", level="WARNING"):
            self.assertEqual(asyncio.run(controller.get_camera_info("ptz")), {})

    def test_aborted_status_request_is_not_reported(self) -> None:
        controller = OnvifController.__new__(OnvifController)

        async def aborted_by_close(request):
            controller.stopping = True
            raise aiohttp.ServerDisconnectedError()

        controller.cams = {
            "ptz": {
                "init": True,
                "ptz": SimpleNamespace(GetStatus=aborted_by_close),
                "status_request": MagicMock(),
            }
        }
        controller.ptz_metrics = {"ptz": PTZMetrics(autotracker_enabled=True)}

        async def status() -> None:
            controller.status_locks = {"ptz": asyncio.Lock()}
            await controller.get_camera_status("ptz")

        with self.assertNoLogs("frigate.ptz.onvif", level="WARNING"):
            asyncio.run(status())


class TestOnvifControllerCloseTwice(unittest.TestCase):
    def test_second_close_returns_at_once(self) -> None:
        controller = _make_controller("front")
        self.addCleanup(_drain, controller)
        controller.close()

        started = time.monotonic()
        controller.close()

        self.assertLess(time.monotonic() - started, 1)
        controller.cams["front"]["onvif"].close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
