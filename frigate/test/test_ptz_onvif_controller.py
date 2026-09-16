"""Tests for the ContinuousMove, Sunba, and preset additions to OnvifController."""

import asyncio
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from frigate.camera import PTZMetrics
from frigate.config import ZoomingModeEnum
from frigate.config.camera.onvif import PtzAutotrackConfig
from frigate.ptz.onvif import OnvifCommandEnum, OnvifController

CAMERA = "front"
PROFILE_TOKEN = "profile_1"


def _make_controller(
    *,
    features: tuple[str, ...] = ("pt", "zoom"),
    sunba: bool = False,
    zooming: ZoomingModeEnum = ZoomingModeEnum.continuous,
) -> OnvifController:
    """Build a controller around hand-made ONVIF service mocks, skipping __init__."""
    controller = OnvifController.__new__(OnvifController)
    controller.config = MagicMock()
    controller.config.cameras = {CAMERA: MagicMock()}
    controller.config.cameras[CAMERA].onvif.autotracking.zooming = zooming
    controller.config.cameras[CAMERA].onvif.autotracking.return_preset = "home"

    ptz = MagicMock()
    ptz.sent_velocities = []

    async def record_move(request):
        ptz.sent_velocities.append(copy.deepcopy(request.Velocity))

    ptz.ContinuousMove = AsyncMock(side_effect=record_move)
    ptz.Stop = AsyncMock()
    ptz.SetPreset = AsyncMock()
    ptz.GetStatus = AsyncMock()

    controller.cams = {
        CAMERA: {
            "init": True,
            "active": False,
            "features": list(features),
            "presets": {"home": "token_home"},
            "profiles": [],
            "sunba_quirks": sunba,
            "ptz": ptz,
            "imaging": None,
            "video_source_token": None,
            "move_request": SimpleNamespace(ProfileToken=PROFILE_TOKEN, Velocity=None),
            "status_request": SimpleNamespace(ProfileToken=PROFILE_TOKEN),
        }
    }
    controller.failed_cams = {}
    controller.ptz_metrics = {CAMERA: PTZMetrics(autotracker_enabled=True)}
    controller.ptz_metrics[CAMERA].frame_time.value = 100.0
    controller.status_locks = {CAMERA: asyncio.Lock()}
    return controller


class _ControllerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.sleep = AsyncMock()
        patcher = patch("frigate.ptz.onvif.asyncio.sleep", self.sleep)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _cam(self, controller: OnvifController) -> dict:
        return controller.cams[CAMERA]

    def _metrics(self, controller: OnvifController) -> PTZMetrics:
        return controller.ptz_metrics[CAMERA]


class TestApplySunbaSpeedSwap(_ControllerTestCase):
    def test_unchanged_without_quirks(self) -> None:
        controller = _make_controller(sunba=False)

        self.assertEqual(
            controller._apply_sunba_speed_swap(CAMERA, 0.5, -0.2), (0.5, -0.2)
        )

    def test_unchanged_when_magnitudes_match(self) -> None:
        controller = _make_controller(sunba=True)

        self.assertEqual(
            controller._apply_sunba_speed_swap(CAMERA, 0.3, -0.3), (0.3, -0.3)
        )

    def test_swaps_magnitudes_and_keeps_directions(self) -> None:
        controller = _make_controller(sunba=True)

        pan, tilt = controller._apply_sunba_speed_swap(CAMERA, 0.5, -0.2)

        self.assertAlmostEqual(pan, 0.2)
        self.assertAlmostEqual(tilt, -0.5)

    def test_swaps_when_both_negative(self) -> None:
        controller = _make_controller(sunba=True)

        pan, tilt = controller._apply_sunba_speed_swap(CAMERA, -0.1, -0.4)

        self.assertAlmostEqual(pan, -0.4)
        self.assertAlmostEqual(tilt, -0.1)

    def test_pure_tilt_becomes_pure_pan(self) -> None:
        controller = _make_controller(sunba=True)

        pan, tilt = controller._apply_sunba_speed_swap(CAMERA, 0.0, 0.3)

        self.assertAlmostEqual(pan, 0.3)
        self.assertAlmostEqual(tilt, 0.0)


class TestMoveContinuousTimed(_ControllerTestCase):
    def test_requires_pt_feature(self) -> None:
        controller = _make_controller(features=("zoom",))

        asyncio.run(controller._move_continuous_timed(CAMERA, 0.3, 0.1, 0.5))

        self._cam(controller)["ptz"].ContinuousMove.assert_not_awaited()
        self.assertFalse(self._cam(controller)["active"])

    def test_skips_when_already_moving(self) -> None:
        controller = _make_controller()
        self._cam(controller)["active"] = True

        asyncio.run(controller._move_continuous_timed(CAMERA, 0.3, 0.1, 0.5))

        self._cam(controller)["ptz"].ContinuousMove.assert_not_awaited()

    def test_sends_velocity_waits_then_stops(self) -> None:
        controller = _make_controller()
        ptz = self._cam(controller)["ptz"]

        asyncio.run(controller._move_continuous_timed(CAMERA, 0.3, -0.2, 0.75))

        self.assertEqual(ptz.sent_velocities, [{"PanTilt": {"x": 0.3, "y": -0.2}}])
        self.sleep.assert_awaited_once_with(0.75)
        ptz.Stop.assert_awaited_once_with(
            {"ProfileToken": PROFILE_TOKEN, "PanTilt": True, "Zoom": True}
        )
        self.assertFalse(self._cam(controller)["active"])
        self.assertEqual(self._metrics(controller).start_time.value, 100.0)
        self.assertEqual(self._metrics(controller).stop_time.value, 0)
        # without quirks the motor state is left to GetStatus polling
        self.assertFalse(self._metrics(controller).motor_stopped.is_set())

    def test_sunba_swaps_speeds_and_marks_motor_stopped(self) -> None:
        controller = _make_controller(sunba=True)
        ptz = self._cam(controller)["ptz"]

        asyncio.run(controller._move_continuous_timed(CAMERA, 0.4, -0.1, 0.5))

        sent = ptz.sent_velocities[0]["PanTilt"]
        self.assertAlmostEqual(sent["x"], 0.1)
        self.assertAlmostEqual(sent["y"], -0.4)
        self.assertTrue(self._metrics(controller).motor_stopped.is_set())
        self.assertEqual(self._metrics(controller).stop_time.value, 100.0)
        self.assertFalse(self._cam(controller)["active"])

    def test_failure_resets_active_without_stopping(self) -> None:
        controller = _make_controller()
        ptz = self._cam(controller)["ptz"]
        ptz.ContinuousMove = AsyncMock(side_effect=RuntimeError("boom"))

        asyncio.run(controller._move_continuous_timed(CAMERA, 0.3, 0.1, 0.5))

        self.assertFalse(self._cam(controller)["active"])
        ptz.Stop.assert_not_awaited()
        self.sleep.assert_not_awaited()


class TestMoveContinuousTimedWithZoom(_ControllerTestCase):
    def test_combines_pan_tilt_and_zoom_in_one_request(self) -> None:
        controller = _make_controller()
        ptz = self._cam(controller)["ptz"]

        asyncio.run(
            controller._move_continuous_timed_with_zoom(CAMERA, 0.5, -0.2, 0.6, 0.3)
        )

        self.assertEqual(
            ptz.sent_velocities,
            [{"PanTilt": {"x": 0.5, "y": -0.2}, "Zoom": {"x": 0.3}}],
        )
        self.sleep.assert_awaited_once_with(0.6)
        ptz.Stop.assert_awaited_once()
        self.assertFalse(self._cam(controller)["active"])

    def test_failure_resets_active(self) -> None:
        controller = _make_controller()
        ptz = self._cam(controller)["ptz"]
        ptz.ContinuousMove = AsyncMock(side_effect=RuntimeError("boom"))

        asyncio.run(
            controller._move_continuous_timed_with_zoom(CAMERA, 0.5, -0.2, 0.6, 0.3)
        )

        self.assertFalse(self._cam(controller)["active"])
        ptz.Stop.assert_not_awaited()

    def test_sunba_sends_pan_tilt_and_zoom_separately(self) -> None:
        controller = _make_controller(sunba=True)
        ptz = self._cam(controller)["ptz"]

        asyncio.run(
            controller._move_continuous_timed_with_zoom(CAMERA, 0.5, -0.2, 0.6, 0.3)
        )

        self.assertEqual(len(ptz.sent_velocities), 2)
        pan_tilt = ptz.sent_velocities[0]["PanTilt"]
        self.assertAlmostEqual(pan_tilt["x"], 0.2)
        self.assertAlmostEqual(pan_tilt["y"], -0.5)
        self.assertNotIn("Zoom", ptz.sent_velocities[0])
        self.assertEqual(ptz.sent_velocities[1], {"Zoom": {"x": 0.3}})
        self.assertEqual(ptz.Stop.await_count, 2)
        self.assertTrue(self._metrics(controller).motor_stopped.is_set())

    def test_sunba_skips_pan_tilt_when_zero(self) -> None:
        controller = _make_controller(sunba=True)
        ptz = self._cam(controller)["ptz"]

        asyncio.run(
            controller._move_continuous_timed_with_zoom(CAMERA, 0.0, 0.0, 0.6, -0.3)
        )

        self.assertEqual(ptz.sent_velocities, [{"Zoom": {"x": -0.3}}])

    def test_sunba_skips_zoom_when_zero(self) -> None:
        controller = _make_controller(sunba=True)
        ptz = self._cam(controller)["ptz"]

        asyncio.run(
            controller._move_continuous_timed_with_zoom(CAMERA, 0.3, 0.3, 0.6, 0.0)
        )

        self.assertEqual(ptz.sent_velocities, [{"PanTilt": {"x": 0.3, "y": 0.3}}])


class TestZoomContinuousTimed(_ControllerTestCase):
    def test_requires_zoom_feature(self) -> None:
        controller = _make_controller(features=("pt",))

        asyncio.run(controller._zoom_continuous_timed(CAMERA, 0.3, 0.5))

        self._cam(controller)["ptz"].ContinuousMove.assert_not_awaited()

    def test_sends_zoom_only_velocity_then_stops(self) -> None:
        controller = _make_controller()
        ptz = self._cam(controller)["ptz"]

        asyncio.run(controller._zoom_continuous_timed(CAMERA, -0.3, 1.5))

        self.assertEqual(ptz.sent_velocities, [{"Zoom": {"x": -0.3}}])
        self.sleep.assert_awaited_once_with(1.5)
        ptz.Stop.assert_awaited_once()
        self.assertFalse(self._cam(controller)["active"])
        self.assertFalse(self._metrics(controller).motor_stopped.is_set())

    def test_sunba_marks_motor_stopped(self) -> None:
        controller = _make_controller(sunba=True)

        asyncio.run(controller._zoom_continuous_timed(CAMERA, 0.3, 0.5))

        self.assertTrue(self._metrics(controller).motor_stopped.is_set())
        self.assertEqual(self._metrics(controller).stop_time.value, 100.0)

    def test_failure_resets_active(self) -> None:
        controller = _make_controller()
        ptz = self._cam(controller)["ptz"]
        ptz.ContinuousMove = AsyncMock(side_effect=RuntimeError("boom"))

        asyncio.run(controller._zoom_continuous_timed(CAMERA, 0.3, 0.5))

        self.assertFalse(self._cam(controller)["active"])
        ptz.Stop.assert_not_awaited()


class TestSetPreset(_ControllerTestCase):
    def test_unknown_preset_is_rejected(self) -> None:
        controller = _make_controller()

        asyncio.run(controller._set_preset(CAMERA, "garage"))

        self._cam(controller)["ptz"].SetPreset.assert_not_awaited()

    def test_updates_existing_preset_token(self) -> None:
        controller = _make_controller()

        asyncio.run(controller._set_preset(CAMERA, "home"))

        self._cam(controller)["ptz"].SetPreset.assert_awaited_once_with(
            {"ProfileToken": PROFILE_TOKEN, "PresetToken": "token_home"}
        )

    def test_camera_error_is_swallowed(self) -> None:
        controller = _make_controller()
        self._cam(controller)["ptz"].SetPreset = AsyncMock(
            side_effect=RuntimeError("boom")
        )

        asyncio.run(controller._set_preset(CAMERA, "home"))


class TestHandleCommandSetReturnPreset(_ControllerTestCase):
    def test_routes_to_configured_return_preset(self) -> None:
        controller = _make_controller()
        controller._set_preset = AsyncMock()

        asyncio.run(
            controller.handle_command_async(
                CAMERA, OnvifCommandEnum.set_return_preset, "return_preset"
            )
        )

        controller._set_preset.assert_awaited_once_with(CAMERA, "home")

    def test_end_to_end_updates_camera_preset(self) -> None:
        controller = _make_controller()

        asyncio.run(
            controller.handle_command_async(
                CAMERA, OnvifCommandEnum.set_return_preset, "return_preset"
            )
        )

        self._cam(controller)["ptz"].SetPreset.assert_awaited_once_with(
            {"ProfileToken": PROFILE_TOKEN, "PresetToken": "token_home"}
        )


def _idle_status(**position) -> SimpleNamespace:
    status = SimpleNamespace(MoveStatus=SimpleNamespace(PanTilt="IDLE", Zoom="IDLE"))
    if position:
        status.Position = SimpleNamespace(**position)
    return status


class TestGetCameraStatusZoomLevel(_ControllerTestCase):
    def _run_status(self, controller: OnvifController, status) -> None:
        self._cam(controller)["ptz"].GetStatus = AsyncMock(return_value=status)
        self._metrics(controller).zoom_level.value = 0.42
        asyncio.run(controller.get_camera_status(CAMERA))

    def test_interpolates_reported_zoom_into_unit_range(self) -> None:
        controller = _make_controller(zooming=ZoomingModeEnum.absolute)
        self._cam(controller)["absolute_zoom_range"] = {
            "XRange": {"Min": 0.0, "Max": 100.0}
        }

        self._run_status(controller, _idle_status(Zoom=SimpleNamespace(x=25.0)))

        self.assertAlmostEqual(self._metrics(controller).zoom_level.value, 0.25)

    def test_keeps_last_value_when_position_missing(self) -> None:
        controller = _make_controller(zooming=ZoomingModeEnum.continuous)
        self._cam(controller)["absolute_zoom_range"] = {
            "XRange": {"Min": 0.0, "Max": 100.0}
        }

        self._run_status(controller, _idle_status())

        self.assertAlmostEqual(self._metrics(controller).zoom_level.value, 0.42)

    def test_keeps_last_value_when_zoom_is_none(self) -> None:
        controller = _make_controller(zooming=ZoomingModeEnum.continuous)
        self._cam(controller)["absolute_zoom_range"] = {
            "XRange": {"Min": 0.0, "Max": 100.0}
        }

        self._run_status(controller, _idle_status(Zoom=None))

        self.assertAlmostEqual(self._metrics(controller).zoom_level.value, 0.42)

    def test_keeps_last_value_without_absolute_range(self) -> None:
        controller = _make_controller(zooming=ZoomingModeEnum.continuous)

        self._run_status(controller, _idle_status(Zoom=SimpleNamespace(x=25.0)))

        self.assertAlmostEqual(self._metrics(controller).zoom_level.value, 0.42)

    def test_malformed_zoom_value_is_ignored(self) -> None:
        controller = _make_controller(zooming=ZoomingModeEnum.absolute)
        self._cam(controller)["absolute_zoom_range"] = {
            "XRange": {"Min": 0.0, "Max": 100.0}
        }

        self._run_status(controller, _idle_status(Zoom=SimpleNamespace(x="bad")))

        self.assertAlmostEqual(self._metrics(controller).zoom_level.value, 0.42)

    def test_zoom_is_not_read_when_zooming_disabled(self) -> None:
        controller = _make_controller(zooming=ZoomingModeEnum.disabled)
        self._cam(controller)["absolute_zoom_range"] = {
            "XRange": {"Min": 0.0, "Max": 100.0}
        }

        self._run_status(controller, _idle_status(Zoom=SimpleNamespace(x=25.0)))

        self.assertAlmostEqual(self._metrics(controller).zoom_level.value, 0.42)


class _Node(dict):
    """Minimal stand-in for zeep objects: attribute and item access on one dict."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def _make_init_controller(
    *,
    autotracking: PtzAutotrackConfig,
    zoom_space: str | None = "zoom-velocity-space",
    relative_pan_tilt_space: str | None = None,
    ptz_config: _Node | None = None,
) -> OnvifController:
    """Build a controller whose ONVIF services are fully mocked for _init_onvif."""
    profile = SimpleNamespace(
        token=PROFILE_TOKEN,
        Name="main",
        VideoEncoderConfiguration=True,
        PTZConfiguration=SimpleNamespace(
            token="ptz-config",
            DefaultContinuousPanTiltVelocitySpace="pt-velocity-space",
            DefaultContinuousZoomVelocitySpace=zoom_space,
            DefaultRelativePanTiltTranslationSpace=relative_pan_tilt_space,
            DefaultRelativeZoomTranslationSpace=None,
            DefaultAbsoluteZoomPositionSpace=None,
            DefaultPTZSpeed=None,
        ),
    )

    media = MagicMock()
    media.GetProfiles = AsyncMock(return_value=[profile])
    media.GetVideoSources = AsyncMock(return_value=[SimpleNamespace(token="vs-1")])

    ptz = MagicMock()
    ptz.create_type = MagicMock(
        side_effect=lambda name: (
            SimpleNamespace(ProfileToken=None, Translation=None, Speed=None)
            if name == "RelativeMove"
            else MagicMock()
        )
    )
    ptz.GetConfigurationOptions = (
        AsyncMock(return_value=ptz_config)
        if ptz_config is not None
        else AsyncMock(side_effect=RuntimeError("unsupported"))
    )
    ptz.GetStatus = AsyncMock(side_effect=RuntimeError("unsupported"))
    ptz.GetPresets = AsyncMock(return_value=[])

    onvif = MagicMock()
    onvif.update_xaddrs = AsyncMock()
    onvif.create_media_service = AsyncMock(return_value=media)
    onvif.create_ptz_service = AsyncMock(return_value=ptz)
    onvif.create_imaging_service = AsyncMock(side_effect=RuntimeError("no imaging"))

    controller = OnvifController.__new__(OnvifController)
    controller.config = MagicMock()
    controller.config.cameras = {CAMERA: MagicMock()}
    controller.config.cameras[CAMERA].onvif.profile = None
    controller.config.cameras[CAMERA].onvif.autotracking = autotracking
    controller.cams = {
        CAMERA: {
            "onvif": onvif,
            "init": False,
            "active": False,
            "features": [],
            "presets": {},
            "profiles": [],
            "sunba_quirks": False,
        }
    }
    controller.failed_cams = {}
    controller.ptz_metrics = {CAMERA: PTZMetrics(autotracker_enabled=True)}
    controller.status_locks = {CAMERA: asyncio.Lock()}
    return controller


def _autotracking(zooming: ZoomingModeEnum) -> PtzAutotrackConfig:
    return PtzAutotrackConfig(enabled=True, enabled_in_config=True, zooming=zooming)


def _fov_ptz_config(*, with_generic_zoom_space: bool) -> _Node:
    spaces = _Node(
        RelativePanTiltTranslationSpace=[
            _Node(
                URI="http://www.onvif.org/ver10/tptz/PanTiltSpaces/TranslationSpaceFov"
            )
        ]
    )
    if with_generic_zoom_space:
        spaces["RelativeZoomTranslationSpace"] = [
            _Node(
                URI="http://www.onvif.org/ver10/tptz/ZoomSpaces/TranslationGenericSpace"
            )
        ]
    return _Node(Spaces=spaces)


class TestInitOnvifContinuousZoomGuard(unittest.TestCase):
    def test_continuous_zoom_kept_when_camera_supports_zoom(self) -> None:
        autotracking = _autotracking(ZoomingModeEnum.continuous)
        controller = _make_init_controller(autotracking=autotracking)

        self.assertTrue(asyncio.run(controller._init_onvif(CAMERA)))

        self.assertEqual(controller.cams[CAMERA]["features"], ["pt", "zoom"])
        self.assertEqual(autotracking.zooming, ZoomingModeEnum.continuous)
        self.assertTrue(controller.cams[CAMERA]["init"])

    def test_continuous_zoom_disabled_without_zoom_support(self) -> None:
        autotracking = _autotracking(ZoomingModeEnum.continuous)
        controller = _make_init_controller(autotracking=autotracking, zoom_space=None)

        self.assertTrue(asyncio.run(controller._init_onvif(CAMERA)))

        self.assertEqual(controller.cams[CAMERA]["features"], ["pt"])
        self.assertEqual(autotracking.zooming, ZoomingModeEnum.disabled)

    def test_disabled_zooming_is_left_alone(self) -> None:
        autotracking = _autotracking(ZoomingModeEnum.disabled)
        controller = _make_init_controller(autotracking=autotracking, zoom_space=None)

        self.assertTrue(asyncio.run(controller._init_onvif(CAMERA)))

        self.assertEqual(autotracking.zooming, ZoomingModeEnum.disabled)


class TestInitOnvifRelativeZoomSpaceGuard(unittest.TestCase):
    def test_missing_relative_zoom_space_keeps_continuous_zoom(self) -> None:
        autotracking = _autotracking(ZoomingModeEnum.continuous)
        controller = _make_init_controller(
            autotracking=autotracking,
            relative_pan_tilt_space="pt-translation-space",
            ptz_config=_fov_ptz_config(with_generic_zoom_space=False),
        )

        self.assertTrue(asyncio.run(controller._init_onvif(CAMERA)))

        self.assertIn("pt-r-fov", controller.cams[CAMERA]["features"])
        self.assertEqual(autotracking.zooming, ZoomingModeEnum.continuous)

    def test_unusable_relative_zoom_space_keeps_continuous_zoom(self) -> None:
        autotracking = _autotracking(ZoomingModeEnum.continuous)
        controller = _make_init_controller(
            autotracking=autotracking,
            relative_pan_tilt_space="pt-translation-space",
            ptz_config=_fov_ptz_config(with_generic_zoom_space=True),
        )

        with self.assertLogs("frigate.ptz.onvif", level="DEBUG") as logs:
            self.assertTrue(asyncio.run(controller._init_onvif(CAMERA)))

        self.assertEqual(autotracking.zooming, ZoomingModeEnum.continuous)
        self.assertTrue(
            any("Relative zoom space not available" in line for line in logs.output)
        )

    def test_unusable_relative_zoom_space_disables_relative_zoom(self) -> None:
        autotracking = _autotracking(ZoomingModeEnum.relative)
        controller = _make_init_controller(
            autotracking=autotracking,
            relative_pan_tilt_space="pt-translation-space",
            ptz_config=_fov_ptz_config(with_generic_zoom_space=True),
        )

        with self.assertLogs("frigate.ptz.onvif", level="WARNING") as logs:
            self.assertTrue(asyncio.run(controller._init_onvif(CAMERA)))

        self.assertEqual(autotracking.zooming, ZoomingModeEnum.disabled)
        self.assertTrue(
            any("Relative zoom not supported" in line for line in logs.output)
        )
