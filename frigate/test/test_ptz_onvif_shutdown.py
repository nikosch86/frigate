"""Tests for a clean ONVIF controller shutdown.

Each ONVIFCamera holds aiohttp sessions, and the controller runs a config poll
that never returns. Left open or pending, they are garbage collected during
shutdown and logged as "Unclosed client session" or "Task was destroyed but it
is pending!" after the log queue has already gone.
"""

import asyncio
import threading
import unittest
from unittest.mock import AsyncMock, MagicMock

from frigate.ptz.onvif import OnvifController


def _drain(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel what a test left on the stopped loop, so it is not logged at exit."""
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
    controller.config_poll = None
    controller.cams = {
        name: {"onvif": MagicMock(close=AsyncMock())} for name in camera_names
    }
    return controller


class TestOnvifControllerClose(unittest.TestCase):
    def _controller(self, *camera_names: str) -> OnvifController:
        controller = _make_controller(*camera_names)
        self.addCleanup(_drain, controller.loop)
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

    def test_camera_without_a_client_is_skipped(self) -> None:
        # a camera whose ONVIF client could not be built has no "onvif" entry
        controller = self._controller("front")
        controller.cams["back"] = {"init": False}

        controller.close()

        controller.cams["front"]["onvif"].close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
