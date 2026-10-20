"""Tests for closing the ONVIF sessions when the controller shuts down.

Each ONVIFCamera holds aiohttp sessions. Left open, they are garbage collected
during shutdown and logged as "Unclosed client session" after the log queue has
already gone.
"""

import asyncio
import threading
import unittest
from unittest.mock import AsyncMock, MagicMock

from frigate.ptz.onvif import OnvifController


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
    def test_closes_every_camera_session(self) -> None:
        controller = _make_controller("front", "back")

        controller.close()

        for cam in controller.cams.values():
            cam["onvif"].close.assert_awaited_once()
        self.assertFalse(controller.loop_thread.is_alive())

    def test_failing_close_does_not_skip_the_others(self) -> None:
        controller = _make_controller("front", "back")
        controller.cams["front"]["onvif"].close.side_effect = RuntimeError("boom")

        controller.close()

        controller.cams["back"]["onvif"].close.assert_awaited_once()
        self.assertFalse(controller.loop_thread.is_alive())

    def test_camera_without_a_client_is_skipped(self) -> None:
        # a camera whose ONVIF client could not be built has no "onvif" entry
        controller = _make_controller("front")
        controller.cams["back"] = {"init": False}

        controller.close()

        controller.cams["front"]["onvif"].close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
