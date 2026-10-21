"""Tests for PTZ move commands that fail, such as when the camera drops off the network.

A failed command used to raise out of the move with the camera still marked as busy,
so every later move was skipped as "already performing an action", the autotracker's
move queue ended, and autotracking stopped moving the camera until a restart.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

import aiohttp
from zeep.exceptions import Fault

from frigate.camera import PTZMetrics
from frigate.ptz.onvif import OnvifController

CAMERA = "ptz"


class _Node(dict):
    """Minimal stand-in for zeep objects: attribute and item access on one dict."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value


def _make_controller() -> OnvifController:
    controller = OnvifController.__new__(OnvifController)
    controller.config = MagicMock()
    ptz = MagicMock()
    ptz.RelativeMove = AsyncMock()
    ptz.AbsoluteMove = AsyncMock()
    ptz.GotoPreset = AsyncMock()
    ptz.ContinuousMove = AsyncMock()
    ptz.Stop = AsyncMock()
    controller.cams = {
        CAMERA: {
            "init": True,
            "active": False,
            "ptz": ptz,
            "features": ["pt", "pt-r-fov", "zoom", "zoom-r", "zoom-a"],
            "presets": {"home": "1"},
            "move_request": MagicMock(ProfileToken="profile_1"),
            "relative_move_request": _Node(
                Translation=_Node(PanTilt=_Node(x=0, y=0)), Speed=None
            ),
            "absolute_move_request": _Node(Speed=None, Position=None),
            "relative_fov_range": {
                "XRange": {"Min": -1.0, "Max": 1.0},
                "YRange": {"Min": -1.0, "Max": 1.0},
            },
            "absolute_zoom_range": {"XRange": {"Min": 0.0, "Max": 1.0}},
            "video_source_token": None,
            "imaging": None,
        }
    }
    controller.ptz_metrics = {CAMERA: PTZMetrics(autotracker_enabled=True)}
    controller.ptz_metrics[CAMERA].frame_time.value = 100.0
    return controller


class TestFailedMoveCommands(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = _make_controller()
        self.cam = self.controller.cams[CAMERA]
        self.metrics = self.controller.ptz_metrics[CAMERA]

    def _assert_move_ended(self) -> None:
        # a move that never happened must not keep the camera busy or moving
        self.assertFalse(self.cam["active"])
        self.assertTrue(self.metrics.motor_stopped.is_set())
        self.assertEqual(self.metrics.stop_time.value, 100.0)

    def test_failed_relative_move_frees_the_camera(self) -> None:
        self.cam["ptz"].RelativeMove.side_effect = aiohttp.ClientConnectionError()

        with self.assertLogs("frigate.ptz.onvif", level="WARNING") as logs:
            asyncio.run(self.controller._move_relative(CAMERA, 0.5, 0.1, 0, 1))

        self._assert_move_ended()
        self.assertTrue(any("RelativeMove for ptz failed" in m for m in logs.output))

    def test_failed_relative_move_resets_its_request(self) -> None:
        # a zoom left in the shared request would ride along with the next move
        self.cam["ptz"].RelativeMove.side_effect = aiohttp.ClientConnectionError()

        with self.assertLogs("frigate.ptz.onvif", level="WARNING"):
            asyncio.run(self.controller._move_relative(CAMERA, 0.5, 0.1, 0.2, 1))

        translation = self.cam["relative_move_request"]["Translation"]
        self.assertNotIn("Zoom", translation)
        self.assertEqual(translation["PanTilt"]["x"], 0)
        self.assertEqual(translation["PanTilt"]["y"], 0)

    def test_next_relative_move_after_a_failure_is_sent(self) -> None:
        self.cam["ptz"].RelativeMove.side_effect = [
            aiohttp.ClientConnectionError(),
            None,
        ]

        async def two_moves() -> None:
            await self.controller._move_relative(CAMERA, 0.5, 0.1, 0, 1)
            await self.controller._move_relative(CAMERA, 0.5, 0.1, 0, 1)

        with self.assertLogs("frigate.ptz.onvif", level="WARNING"):
            asyncio.run(two_moves())

        self.assertEqual(self.cam["ptz"].RelativeMove.await_count, 2)
        self.assertFalse(self.cam["active"])

    def test_failed_preset_move_frees_the_camera(self) -> None:
        # raising here ended the autotracker thread's maintenance for every camera
        self.cam["ptz"].GotoPreset.side_effect = Fault("not reachable")

        with self.assertLogs("frigate.ptz.onvif", level="WARNING") as logs:
            asyncio.run(self.controller._move_to_preset(CAMERA, "home"))

        self._assert_move_ended()
        self.assertTrue(any("GotoPreset for ptz failed" in m for m in logs.output))

    def test_failed_absolute_zoom_frees_the_camera(self) -> None:
        self.cam["ptz"].AbsoluteMove.side_effect = TimeoutError()

        with self.assertLogs("frigate.ptz.onvif", level="WARNING") as logs:
            asyncio.run(self.controller._zoom_absolute(CAMERA, 0.5, 1))

        self._assert_move_ended()
        self.assertTrue(any("AbsoluteMove for ptz failed" in m for m in logs.output))

    # the timed ContinuousMove variants, called with a zero duration
    CONTINUOUS_MOVES = (
        ("_move_continuous_timed", (0.5, 0.1, 0)),
        ("_move_continuous_timed_with_zoom", (0.5, 0.1, 0, 0.2)),
        ("_zoom_continuous_timed", (0.2, 0)),
    )

    def test_failed_continuous_move_frees_the_camera(self) -> None:
        for method, args in self.CONTINUOUS_MOVES:
            with self.subTest(method=method):
                self.setUp()
                self.cam["ptz"].ContinuousMove.side_effect = TimeoutError()

                with self.assertLogs("frigate.ptz.onvif", level="WARNING"):
                    asyncio.run(getattr(self.controller, method)(CAMERA, *args))

                self._assert_move_ended()
                self.cam["ptz"].Stop.assert_not_awaited()

    def test_failed_stop_leaves_the_motor_to_the_status_poll(self) -> None:
        # the camera received the move and may still be moving
        for method, args in self.CONTINUOUS_MOVES:
            with self.subTest(method=method):
                self.setUp()
                self.cam["ptz"].Stop.side_effect = aiohttp.ClientConnectionError()

                with self.assertLogs("frigate.ptz.onvif", level="WARNING") as logs:
                    asyncio.run(getattr(self.controller, method)(CAMERA, *args))

                self.assertFalse(self.cam["active"])
                self.assertFalse(self.metrics.motor_stopped.is_set())
                self.assertEqual(self.metrics.stop_time.value, 0)
                self.assertTrue(
                    any(f"Stop for {CAMERA} failed" in m for m in logs.output)
                )

    def test_successful_moves_still_leave_the_motor_to_the_status_poll(self) -> None:
        # only the camera's status may end a move that was really sent
        asyncio.run(self.controller._move_relative(CAMERA, 0.5, 0.1, 0, 1))

        self.assertFalse(self.cam["active"])
        self.assertFalse(self.metrics.motor_stopped.is_set())
        self.assertEqual(self.metrics.stop_time.value, 0)


if __name__ == "__main__":
    unittest.main()
