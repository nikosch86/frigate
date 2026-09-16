"""Tests for the ContinuousMove and continuous-zoom paths of PtzAutoTracker."""

import asyncio
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from frigate.camera import PTZMetrics
from frigate.config import ZoomingModeEnum
from frigate.config.camera.onvif import PtzAutotrackConfig
from frigate.const import AUTOTRACKING_MAX_AREA_RATIO
from frigate.ptz.autotrack import PtzAutoTracker

CAMERA = "front"
ZOOM_FACTOR = 0.3
MAX_TARGET_BOX = AUTOTRACKING_MAX_AREA_RATIO ** (1 / ZOOM_FACTOR)


def _make_tracker(
    *,
    features: tuple[str, ...] = ("pt", "zoom"),
    zooming: ZoomingModeEnum = ZoomingModeEnum.continuous,
    sunba: bool = False,
    frame_shape: tuple[int, int] = (720, 1280),
    fps: int = 5,
    **autotracking_overrides,
) -> PtzAutoTracker:
    """Build a tracker with a real autotracking config and mocked ONVIF layer."""
    tracker = PtzAutoTracker.__new__(PtzAutoTracker)

    autotracking = PtzAutotrackConfig(
        enabled=True,
        enabled_in_config=True,
        zooming=zooming,
        zoom_factor=ZOOM_FACTOR,
        return_preset="home",
        **autotracking_overrides,
    )
    camera_config = MagicMock()
    camera_config.name = CAMERA
    camera_config.enabled = True
    camera_config.frame_shape = frame_shape
    camera_config.detect.fps = fps
    camera_config.detect.height = frame_shape[0]
    camera_config.detect.width = frame_shape[1]
    camera_config.onvif.autotracking = autotracking
    camera_config.onvif.sunba_quirks = sunba

    tracker.config = MagicMock()
    tracker.config.cameras = {CAMERA: camera_config}

    onvif = MagicMock()
    onvif.cams = {CAMERA: {"init": True, "features": list(features)}}
    onvif.loop = MagicMock()
    for name in (
        "get_camera_status",
        "_init_onvif",
        "_move_to_preset",
        "_move_relative",
        "_move_continuous_timed",
        "_zoom_continuous_timed",
        "_zoom_absolute",
    ):
        setattr(onvif, name, AsyncMock())
    tracker.onvif = onvif

    tracker.ptz_metrics = {CAMERA: PTZMetrics(autotracker_enabled=True)}
    tracker.dispatcher = MagicMock()
    tracker.stop_event = MagicMock()

    for name in (
        "tracked_object",
        "tracked_object_history",
        "tracked_object_metrics",
        "object_types",
        "required_zones",
        "move_queues",
        "move_queue_locks",
        "move_threads",
        "autotracker_init",
        "move_metrics",
        "calibrating",
        "intercept",
        "move_coefficients",
        "zoom_time",
        "zoom_factor",
        "movement_mode",
        "continuous_speed",
        "continuous_zoom_speed",
        "estimated_zoom_position",
        "native_zoom_range",
    ):
        setattr(tracker, name, {})

    tracker.zoom_factor[CAMERA] = ZOOM_FACTOR
    tracker.intercept[CAMERA] = None
    tracker.move_metrics[CAMERA] = []
    tracker.calibrating[CAMERA] = False
    tracker.autotracker_init[CAMERA] = True
    tracker.tracked_object[CAMERA] = None
    tracker.tracked_object_history[CAMERA] = deque(maxlen=8)
    tracker.tracked_object_metrics[CAMERA] = {"max_target_box": MAX_TARGET_BOX}
    return tracker


def _config(tracker: PtzAutoTracker) -> PtzAutotrackConfig:
    return tracker.config.cameras[CAMERA].onvif.autotracking


def _metrics(tracker: PtzAutoTracker) -> PTZMetrics:
    return tracker.ptz_metrics[CAMERA]


def _obj(box, frame_time: float = 10.0, obj_id: str = "obj-1") -> SimpleNamespace:
    centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    return SimpleNamespace(
        obj_data={
            "id": obj_id,
            "box": list(box),
            "centroid": centroid,
            "frame_time": frame_time,
        }
    )


def _run_setup(tracker: PtzAutoTracker) -> None:
    # feature detection only runs when setup initializes the camera itself,
    # so start uninitialized and let the mocked init flip the flag
    cam = tracker.onvif.cams[CAMERA]
    cam["init"] = False

    async def init_onvif(camera):
        cam["init"] = True
        return True

    tracker.onvif._init_onvif = AsyncMock(side_effect=init_onvif)
    tracker.onvif.get_service_capabilities = AsyncMock(return_value=True)

    # the move queue consumer is normally scheduled on the ONVIF loop thread;
    # close the coroutine instead so nothing runs in the background
    with patch(
        "frigate.ptz.autotrack.asyncio.run_coroutine_threadsafe",
        side_effect=lambda coro, loop: coro.close(),
    ):
        asyncio.run(tracker._autotracker_setup(tracker.config.cameras[CAMERA], CAMERA))


class TestAutotrackerSetupMovementMode(unittest.TestCase):
    def test_prefers_relative_fov_when_supported(self) -> None:
        tracker = _make_tracker(
            features=("pt", "pt-r-fov"), zooming=ZoomingModeEnum.disabled
        )

        _run_setup(tracker)

        self.assertEqual(tracker.movement_mode[CAMERA], "relative_fov")
        self.assertNotIn(CAMERA, tracker.continuous_speed)
        self.assertTrue(tracker.autotracker_init[CAMERA])
        tracker.dispatcher.publish.assert_called_once_with(
            f"{CAMERA}/ptz_autotracker/active", "OFF", retain=False
        )

    def test_falls_back_to_continuous_timed(self) -> None:
        tracker = _make_tracker(
            features=("pt",),
            zooming=ZoomingModeEnum.disabled,
            continuous_speed=3.5,
            continuous_zoom_speed=0.5,
        )

        _run_setup(tracker)

        self.assertEqual(tracker.movement_mode[CAMERA], "continuous_timed")
        self.assertEqual(tracker.continuous_speed[CAMERA], 3.5)
        self.assertEqual(tracker.continuous_zoom_speed[CAMERA], 0.5)
        self.assertTrue(_config(tracker).enabled)

    def test_disables_autotracking_without_movement_support(self) -> None:
        tracker = _make_tracker(features=("zoom",), zooming=ZoomingModeEnum.disabled)

        _run_setup(tracker)

        self.assertNotIn(CAMERA, tracker.movement_mode)
        self.assertFalse(_config(tracker).enabled)
        self.assertFalse(_metrics(tracker).autotracker_enabled.value)

    def test_continuous_zoom_without_zoom_feature_keeps_moving(self) -> None:
        tracker = _make_tracker(features=("pt",), zooming=ZoomingModeEnum.continuous)

        _run_setup(tracker)

        self.assertEqual(tracker.movement_mode[CAMERA], "continuous_timed")

    def test_absolute_and_relative_zoom_modes_are_logged_only(self) -> None:
        for zooming, feature in (
            (ZoomingModeEnum.absolute, "zoom-a"),
            (ZoomingModeEnum.absolute, None),
            (ZoomingModeEnum.relative, "zoom-r"),
            (ZoomingModeEnum.relative, None),
        ):
            with self.subTest(zooming=zooming, feature=feature):
                features = ("pt", feature) if feature else ("pt",)
                tracker = _make_tracker(features=features, zooming=zooming)
                _metrics(tracker).zoom_level.value = 0.4

                _run_setup(tracker)

                self.assertEqual(tracker.movement_mode[CAMERA], "continuous_timed")
                self.assertEqual(_config(tracker).zooming, zooming)


class TestAutotrackerSetupZoomLevelDetection(unittest.TestCase):
    def test_reported_zoom_level_needs_no_estimate(self) -> None:
        tracker = _make_tracker()
        _metrics(tracker).zoom_level.value = 0.3

        _run_setup(tracker)

        self.assertNotIn(CAMERA, tracker.estimated_zoom_position)
        self.assertNotIn(CAMERA, tracker.native_zoom_range)
        self.assertEqual(_config(tracker).zooming, ZoomingModeEnum.continuous)

    def test_continuous_without_report_uses_default_range(self) -> None:
        tracker = _make_tracker()

        _run_setup(tracker)

        self.assertEqual(tracker.native_zoom_range[CAMERA], (1.0, 30.0))
        self.assertEqual(tracker.estimated_zoom_position[CAMERA], 0.5)
        self.assertEqual(_metrics(tracker).min_zoom.value, 0.0)
        self.assertEqual(_metrics(tracker).max_zoom.value, 1.0)

    def test_continuous_without_report_uses_configured_range_and_preset(
        self,
    ) -> None:
        tracker = _make_tracker(assumed_zoom_range=[2, 12], preset_zoom_level=4)

        _run_setup(tracker)

        self.assertEqual(tracker.native_zoom_range[CAMERA], (2.0, 12.0))
        self.assertAlmostEqual(tracker.estimated_zoom_position[CAMERA], 0.2)

    def test_absolute_without_report_disables_zooming(self) -> None:
        tracker = _make_tracker(
            features=("pt", "zoom-a"), zooming=ZoomingModeEnum.absolute
        )

        _run_setup(tracker)

        self.assertEqual(_config(tracker).zooming, ZoomingModeEnum.disabled)
        self.assertNotIn(CAMERA, tracker.estimated_zoom_position)


class TestZoomConversion(unittest.TestCase):
    def test_passthrough_without_native_range(self) -> None:
        tracker = _make_tracker()

        self.assertEqual(tracker._native_to_normalized_zoom(CAMERA, 0.7), 0.7)
        self.assertEqual(tracker._normalized_to_native_zoom(CAMERA, 0.7), 0.7)

    def test_native_to_normalized_maps_range_to_unit_interval(self) -> None:
        tracker = _make_tracker()
        tracker.native_zoom_range[CAMERA] = (1.0, 30.0)

        self.assertAlmostEqual(tracker._native_to_normalized_zoom(CAMERA, 1.0), 0.0)
        self.assertAlmostEqual(tracker._native_to_normalized_zoom(CAMERA, 30.0), 1.0)
        self.assertAlmostEqual(tracker._native_to_normalized_zoom(CAMERA, 15.5), 0.5)

    def test_native_to_normalized_clips_outside_range(self) -> None:
        tracker = _make_tracker()
        tracker.native_zoom_range[CAMERA] = (1.0, 30.0)

        self.assertEqual(tracker._native_to_normalized_zoom(CAMERA, 0.5), 0.0)
        self.assertEqual(tracker._native_to_normalized_zoom(CAMERA, 45.0), 1.0)

    def test_normalized_to_native_maps_back(self) -> None:
        tracker = _make_tracker()
        tracker.native_zoom_range[CAMERA] = (1.0, 30.0)

        self.assertAlmostEqual(tracker._normalized_to_native_zoom(CAMERA, 0.0), 1.0)
        self.assertAlmostEqual(tracker._normalized_to_native_zoom(CAMERA, 1.0), 30.0)
        self.assertAlmostEqual(tracker._normalized_to_native_zoom(CAMERA, 0.5), 15.5)

    def test_round_trip(self) -> None:
        tracker = _make_tracker()
        tracker.native_zoom_range[CAMERA] = (2.0, 12.0)

        normalized = tracker._native_to_normalized_zoom(CAMERA, 7.0)

        self.assertAlmostEqual(
            tracker._normalized_to_native_zoom(CAMERA, normalized), 7.0
        )


class TestCalculateContinuousMoveParams(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = _make_tracker()
        self.tracker.continuous_speed[CAMERA] = 2.0

    def test_skips_tiny_moves(self) -> None:
        self.assertEqual(
            self.tracker._calculate_continuous_move_params(CAMERA, 0.02, 0.1),
            (0.0, 0.0, 0.0),
        )

    def test_small_correction_uses_proportional_velocity(self) -> None:
        pan, tilt, duration = self.tracker._calculate_continuous_move_params(
            CAMERA, 0.1, 0.0
        )

        self.assertAlmostEqual(pan, 0.15)
        self.assertAlmostEqual(tilt, 0.0)
        self.assertAlmostEqual(duration, 0.1 / (2.0 * 0.15))

    def test_medium_move_uses_default_velocity(self) -> None:
        pan, tilt, duration = self.tracker._calculate_continuous_move_params(
            CAMERA, -0.5, 0.0
        )

        self.assertAlmostEqual(pan, -0.3)
        self.assertAlmostEqual(duration, 0.5 / (2.0 * 0.3))

    def test_large_moves_use_faster_velocity(self) -> None:
        pan_large, _, duration_large = self.tracker._calculate_continuous_move_params(
            CAMERA, 0.8, 0.0
        )
        pan_huge, _, duration_huge = self.tracker._calculate_continuous_move_params(
            CAMERA, 0.95, 0.0
        )

        self.assertAlmostEqual(pan_large, 0.4)
        self.assertAlmostEqual(duration_large, 1.0)
        self.assertAlmostEqual(pan_huge, 0.5)
        self.assertAlmostEqual(duration_huge, 0.95)

    def test_tilt_is_scaled_down_but_keeps_sign(self) -> None:
        pan, tilt, duration = self.tracker._calculate_continuous_move_params(
            CAMERA, 0.0, -1.0
        )

        self.assertAlmostEqual(pan, 0.0)
        self.assertAlmostEqual(tilt, -0.15)
        self.assertAlmostEqual(duration, 0.1 / (2.0 * 0.15))

    def test_duration_is_clamped(self) -> None:
        self.tracker.continuous_speed[CAMERA] = 0.1
        _, _, slow = self.tracker._calculate_continuous_move_params(CAMERA, 0.5, 0.0)
        self.tracker.continuous_speed[CAMERA] = 10.0
        _, _, fast = self.tracker._calculate_continuous_move_params(CAMERA, 0.06, 0.0)

        self.assertEqual(slow, 1.25)
        self.assertEqual(fast, 0.1)

    def test_diagonal_move_splits_velocity_by_direction(self) -> None:
        pan, tilt, _ = self.tracker._calculate_continuous_move_params(CAMERA, 0.3, 3.0)

        self.assertAlmostEqual(np.hypot(pan, tilt), 0.3)
        self.assertAlmostEqual(pan, tilt)


class TestCalculateContinuousZoomParams(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = _make_tracker()
        self.tracker.continuous_zoom_speed[CAMERA] = 1.0

    def test_skips_tiny_zoom(self) -> None:
        self.assertEqual(
            self.tracker._calculate_continuous_zoom_params(CAMERA, 0.005), (0.0, 0.0)
        )

    def test_zoom_in_uses_positive_velocity(self) -> None:
        velocity, duration = self.tracker._calculate_continuous_zoom_params(CAMERA, 0.3)

        self.assertAlmostEqual(velocity, 0.3)
        self.assertAlmostEqual(duration, 1.0)

    def test_zoom_out_uses_negative_velocity(self) -> None:
        velocity, duration = self.tracker._calculate_continuous_zoom_params(
            CAMERA, -0.15
        )

        self.assertAlmostEqual(velocity, -0.3)
        self.assertAlmostEqual(duration, 0.5)

    def test_duration_is_clamped(self) -> None:
        _, short = self.tracker._calculate_continuous_zoom_params(CAMERA, 0.02)
        self.tracker.continuous_zoom_speed[CAMERA] = 0.1
        _, long = self.tracker._calculate_continuous_zoom_params(CAMERA, 1.0)

        self.assertEqual(short, 0.2)
        self.assertEqual(long, 2.0)


class TestPredictAreaAfterTime(unittest.TestCase):
    def test_returns_zero_without_coefficients(self) -> None:
        tracker = _make_tracker()

        self.assertEqual(tracker._predict_area_after_time(CAMERA, 0.5), 0)


class TestDistanceThreshold(unittest.TestCase):
    OBJ = _obj([100, 100, 200, 150])

    def test_relative_fov_keeps_original_scaling(self) -> None:
        tracker = _make_tracker()
        tracker.movement_mode[CAMERA] = "relative_fov"

        self.assertAlmostEqual(
            tracker._get_distance_threshold(CAMERA, self.OBJ), 136.30, places=2
        )

    def test_continuous_widens_dead_zone_and_dampens_scaling(self) -> None:
        tracker = _make_tracker()
        tracker.movement_mode[CAMERA] = "continuous_timed"

        self.assertAlmostEqual(
            tracker._get_distance_threshold(CAMERA, self.OBJ), 323.80, places=2
        )

    def test_continuous_keeps_calibrated_percentage(self) -> None:
        weights = ["0", "1", "0.5", "0.1", "0.2", "1"]
        tracker = _make_tracker(movement_weights=weights)
        tracker.movement_mode[CAMERA] = "continuous_timed"

        tracker.tracked_object_metrics[CAMERA]["valid_velocity"] = True
        self.assertAlmostEqual(
            tracker._get_distance_threshold(CAMERA, self.OBJ), 259.04, places=2
        )

        tracker.tracked_object_metrics[CAMERA]["valid_velocity"] = False
        self.assertAlmostEqual(
            tracker._get_distance_threshold(CAMERA, self.OBJ), 97.14, places=2
        )


class TestShouldZoomIn(unittest.TestCase):
    SMALL_CENTERED = [600, 320, 680, 400]
    FULL_FRAME = [0, 0, 1280, 720]

    def _tracker(self, zooming=ZoomingModeEnum.continuous) -> PtzAutoTracker:
        tracker = _make_tracker(zooming=zooming)
        tracker.tracked_object_metrics[CAMERA]["velocity"] = np.zeros((4,))
        return tracker

    def test_zooms_in_on_small_centered_object(self) -> None:
        tracker = self._tracker()
        tracker.tracked_object_metrics[CAMERA]["target_box"] = 0.05
        tracker.estimated_zoom_position[CAMERA] = 0.5

        self.assertTrue(
            tracker._should_zoom_in(
                CAMERA, _obj(self.SMALL_CENTERED), self.SMALL_CENTERED, 0
            )
        )

    def test_falls_back_to_object_velocity_before_metrics_exist(self) -> None:
        tracker = self._tracker()
        del tracker.tracked_object_metrics[CAMERA]["velocity"]
        tracker.tracked_object_metrics[CAMERA]["target_box"] = 0.05
        tracker.estimated_zoom_position[CAMERA] = 0.5
        obj = _obj(self.SMALL_CENTERED)
        obj.obj_data["estimate_velocity"] = np.zeros((2, 2))

        self.assertTrue(tracker._should_zoom_in(CAMERA, obj, self.SMALL_CENTERED, 0))

    def test_estimated_max_zoom_blocks_zoom_in(self) -> None:
        tracker = self._tracker()
        tracker.tracked_object_metrics[CAMERA]["target_box"] = 0.05
        tracker.estimated_zoom_position[CAMERA] = 0.95
        tracker.native_zoom_range[CAMERA] = (1.0, 30.0)

        self.assertIsNone(
            tracker._should_zoom_in(
                CAMERA,
                _obj(self.SMALL_CENTERED),
                self.SMALL_CENTERED,
                0,
                debug_zooming=True,
            )
        )

    def test_zooms_out_when_object_fills_frame(self) -> None:
        tracker = self._tracker()
        tracker.estimated_zoom_position[CAMERA] = 0.5

        self.assertFalse(
            tracker._should_zoom_in(CAMERA, _obj(self.FULL_FRAME), self.FULL_FRAME, 0)
        )

    def test_estimated_min_zoom_blocks_zoom_out(self) -> None:
        tracker = self._tracker()
        tracker.estimated_zoom_position[CAMERA] = 0.05

        self.assertIsNone(
            tracker._should_zoom_in(CAMERA, _obj(self.FULL_FRAME), self.FULL_FRAME, 0)
        )

    def test_reported_zoom_level_is_used_without_estimate(self) -> None:
        tracker = self._tracker(zooming=ZoomingModeEnum.absolute)
        tracker.tracked_object_metrics[CAMERA]["target_box"] = 0.05
        _metrics(tracker).max_zoom.value = 1.0

        _metrics(tracker).zoom_level.value = 0.5
        self.assertTrue(
            tracker._should_zoom_in(
                CAMERA, _obj(self.SMALL_CENTERED), self.SMALL_CENTERED, 0
            )
        )

        _metrics(tracker).zoom_level.value = 1.0
        self.assertIsNone(
            tracker._should_zoom_in(
                CAMERA, _obj(self.SMALL_CENTERED), self.SMALL_CENTERED, 0
            )
        )


class TestGetZoomAmountContinuous(unittest.TestCase):
    OBJ = _obj([600, 320, 680, 400])

    def _tracker(self, decision) -> PtzAutoTracker:
        tracker = _make_tracker()
        tracker._should_zoom_in = MagicMock(return_value=decision)
        return tracker

    def _zoom(self, tracker: PtzAutoTracker) -> float:
        return tracker._get_zoom_amount(CAMERA, self.OBJ, self.OBJ.obj_data["box"], 0)

    def test_no_decision_means_no_zoom(self) -> None:
        self.assertEqual(self._zoom(self._tracker(None)), 0)

    def test_initial_zoom_uses_fixed_step(self) -> None:
        self.assertAlmostEqual(self._zoom(self._tracker(True)), 0.3)
        self.assertAlmostEqual(self._zoom(self._tracker(False)), -0.3)

    def test_zoom_in_scales_with_remaining_area(self) -> None:
        tracker = self._tracker(True)
        tracker.tracked_object_metrics[CAMERA]["target_box"] = 0.5 * MAX_TARGET_BOX

        self.assertAlmostEqual(self._zoom(tracker), 0.25)

    def test_zoom_in_is_capped(self) -> None:
        tracker = self._tracker(True)
        tracker.tracked_object_metrics[CAMERA]["target_box"] = 0.0

        self.assertAlmostEqual(self._zoom(tracker), 0.5)

    def test_zoom_out_scales_with_excess_area(self) -> None:
        tracker = self._tracker(False)
        tracker.tracked_object_metrics[CAMERA]["target_box"] = 1.2 * MAX_TARGET_BOX

        self.assertAlmostEqual(self._zoom(tracker), -0.1)

    def test_zoom_out_is_capped(self) -> None:
        tracker = self._tracker(False)
        tracker.tracked_object_metrics[CAMERA]["target_box"] = 3 * MAX_TARGET_BOX

        self.assertAlmostEqual(self._zoom(tracker), -0.5)


class TestAutotrackMovePtzPrediction(unittest.TestCase):
    def _tracker(self, mode: str) -> PtzAutoTracker:
        tracker = _make_tracker(frame_shape=(1000, 1000), fps=5)
        tracker.movement_mode[CAMERA] = mode
        tracker.continuous_speed[CAMERA] = 2.0
        tracker.tracked_object_metrics[CAMERA].update(
            {"velocity": np.array([10.0, 0.0, 10.0, 0.0]), "valid_velocity": True}
        )
        tracker._get_zoom_amount = MagicMock(return_value=0)
        tracker._enqueue_move = MagicMock()
        return tracker

    def _enqueued(self, tracker: PtzAutoTracker) -> tuple[float, float]:
        tracker._enqueue_move.assert_called_once()
        _, frame_time, pan, tilt, zoom = tracker._enqueue_move.call_args.args
        self.assertEqual(frame_time, 10.0)
        self.assertEqual(zoom, 0)
        return pan, tilt

    def test_continuous_leads_moving_object(self) -> None:
        tracker = self._tracker("continuous_timed")

        tracker._autotrack_move_ptz(CAMERA, _obj([600, 450, 700, 550]))

        pan, tilt = self._enqueued(tracker)
        self.assertAlmostEqual(pan, 0.346)
        self.assertAlmostEqual(tilt, 0.0)

    def test_continuous_does_not_predict_at_center(self) -> None:
        tracker = self._tracker("continuous_timed")

        tracker._autotrack_move_ptz(CAMERA, _obj([450, 450, 550, 550]))

        pan, tilt = self._enqueued(tracker)
        self.assertAlmostEqual(pan, 0.0)
        self.assertAlmostEqual(tilt, 0.0)

    def test_relative_fov_without_weights_uses_raw_centroid(self) -> None:
        tracker = self._tracker("relative_fov")

        tracker._autotrack_move_ptz(CAMERA, _obj([600, 450, 700, 550]))

        pan, tilt = self._enqueued(tracker)
        self.assertAlmostEqual(pan, 0.3)
        self.assertAlmostEqual(tilt, 0.0)


class TestAutotrackObjectSettling(unittest.TestCase):
    CENTERED = [600, 320, 680, 400]
    NEAR_LEFT_EDGE = [10, 320, 90, 400]

    def _tracker(self, mode: str = "continuous_timed") -> PtzAutoTracker:
        tracker = _make_tracker(fps=5)
        tracker.movement_mode[CAMERA] = mode
        tracker.tracked_object[CAMERA] = _obj(self.CENTERED, frame_time=1.0)
        tracker.tracked_object_history[CAMERA].append({"frame_time": 1.0})
        tracker.tracked_object_metrics[CAMERA]["below_distance_threshold"] = False
        _metrics(tracker).start_time.value = 5.0
        _metrics(tracker).stop_time.value = 6.0
        tracker._calculate_tracked_object_metrics = MagicMock()
        tracker._autotrack_move_ptz = MagicMock()
        tracker._autotrack_move_zoom_only = MagicMock()
        return tracker

    def test_skips_frames_during_normal_settling(self) -> None:
        tracker = self._tracker()

        tracker.autotrack_object(CAMERA, _obj(self.CENTERED, frame_time=6.3))

        tracker._autotrack_move_ptz.assert_not_called()
        tracker._autotrack_move_zoom_only.assert_not_called()

    def test_moves_after_normal_settling(self) -> None:
        tracker = self._tracker()

        tracker.autotrack_object(CAMERA, _obj(self.CENTERED, frame_time=6.7))

        tracker._autotrack_move_ptz.assert_called_once()

    def test_object_near_edge_settles_faster(self) -> None:
        tracker = self._tracker()

        tracker.autotrack_object(CAMERA, _obj(self.NEAR_LEFT_EDGE, frame_time=6.3))

        tracker._autotrack_move_ptz.assert_called_once()

    def test_object_near_edge_still_waits_one_frame(self) -> None:
        tracker = self._tracker()

        tracker.autotrack_object(CAMERA, _obj(self.NEAR_LEFT_EDGE, frame_time=6.1))

        tracker._autotrack_move_ptz.assert_not_called()

    def test_relative_fov_does_not_settle(self) -> None:
        tracker = self._tracker(mode="relative_fov")

        tracker.autotrack_object(CAMERA, _obj(self.CENTERED, frame_time=6.3))

        tracker._autotrack_move_ptz.assert_called_once()

    def test_centered_object_only_zooms_after_settling(self) -> None:
        tracker = self._tracker()
        tracker.tracked_object_metrics[CAMERA]["below_distance_threshold"] = True

        tracker.autotrack_object(CAMERA, _obj(self.CENTERED, frame_time=6.7))

        tracker._autotrack_move_zoom_only.assert_called_once()
        tracker._autotrack_move_ptz.assert_not_called()


class TestCameraMaintenanceZoomReset(unittest.TestCase):
    def _tracker(self, **overrides) -> PtzAutoTracker:
        tracker = _make_tracker(timeout=10, **overrides)
        tracker.native_zoom_range[CAMERA] = (1.0, 30.0)
        tracker.estimated_zoom_position[CAMERA] = 0.9
        tracker.tracked_object_history[CAMERA].append({"frame_time": 0.0})
        _metrics(tracker).frame_time.value = 100.0
        return tracker

    def test_returns_to_preset_and_resets_to_preset_zoom(self) -> None:
        tracker = self._tracker(assumed_zoom_range=[1, 30], preset_zoom_level=10)

        asyncio.run(tracker.camera_maintenance(CAMERA))

        tracker.onvif._move_to_preset.assert_awaited_once_with(CAMERA, "home")
        self.assertAlmostEqual(tracker.estimated_zoom_position[CAMERA], 9 / 29)
        self.assertTrue(_metrics(tracker).reset.is_set())
        tracker.dispatcher.publish.assert_called_once_with(
            f"{CAMERA}/ptz_autotracker/active", "OFF", retain=False
        )

    def test_resets_to_middle_without_preset_level(self) -> None:
        tracker = self._tracker()

        asyncio.run(tracker.camera_maintenance(CAMERA))

        self.assertEqual(tracker.estimated_zoom_position[CAMERA], 0.5)


class TestCalibrateCameraContinuousZoom(unittest.TestCase):
    def _tracker(self, **overrides) -> PtzAutoTracker:
        tracker = _make_tracker(**overrides)
        tracker.estimated_zoom_position[CAMERA] = 0.5
        tracker._calculate_move_coefficients = MagicMock()

        async def release_motor(camera):
            _metrics(tracker).motor_stopped.set()

        tracker.onvif.get_camera_status = AsyncMock(side_effect=release_motor)
        return tracker

    def test_skips_zoom_calibration_with_configured_range(self) -> None:
        tracker = self._tracker(assumed_zoom_range=[1, 30], preset_zoom_level=10)
        tracker.native_zoom_range[CAMERA] = (1.0, 30.0)

        asyncio.run(tracker._calibrate_camera(CAMERA))

        tracker.onvif._zoom_absolute.assert_not_awaited()
        self.assertEqual(_metrics(tracker).min_zoom.value, 0.0)
        self.assertEqual(_metrics(tracker).max_zoom.value, 1.0)
        self.assertFalse(tracker.calibrating[CAMERA])
        tracker._calculate_move_coefficients.assert_called_once_with(CAMERA, True)

    def test_skips_zoom_calibration_with_default_range(self) -> None:
        tracker = self._tracker()

        asyncio.run(tracker._calibrate_camera(CAMERA))

        tracker.onvif._zoom_absolute.assert_not_awaited()
        self.assertEqual(_metrics(tracker).max_zoom.value, 1.0)

    def test_absolute_zoom_is_still_calibrated(self) -> None:
        tracker = self._tracker(zooming=ZoomingModeEnum.absolute)
        del tracker.estimated_zoom_position[CAMERA]
        tracker.onvif.cams[CAMERA]["absolute_zoom_range"] = {
            "XRange": {"Min": 0.0, "Max": 1.0}
        }

        async def zoom_to(camera, zoom, speed):
            _metrics(tracker).zoom_level.value = zoom

        tracker.onvif._zoom_absolute = AsyncMock(side_effect=zoom_to)

        asyncio.run(tracker._calibrate_camera(CAMERA))

        self.assertEqual(tracker.onvif._zoom_absolute.await_count, 3)
        self.assertEqual(_metrics(tracker).min_zoom.value, 0.0)
        self.assertEqual(_metrics(tracker).max_zoom.value, 1.0)


class TestProcessMoveQueueContinuous(unittest.TestCase):
    def _run_queue(self, tracker: PtzAutoTracker, *moves) -> None:
        # one loop iteration per queued move, then the stop flag ends the loop
        tracker.stop_event.is_set.side_effect = [False] * len(moves) + [True]
        tracker._calculate_move_coefficients = MagicMock()

        async def run():
            tracker.move_queues[CAMERA] = asyncio.Queue()
            tracker.move_queue_locks[CAMERA] = asyncio.Lock()
            for move in moves:
                tracker.move_queues[CAMERA].put_nowait(move)
            await tracker._process_move_queue(CAMERA)

        asyncio.run(run())

    def _tracker(self, mode: str = "continuous_timed", **kwargs) -> PtzAutoTracker:
        tracker = _make_tracker(**kwargs)
        tracker.movement_mode[CAMERA] = mode
        tracker.continuous_speed[CAMERA] = 2.0
        tracker.continuous_zoom_speed[CAMERA] = 1.0

        # every move starts the motor and the next status poll stops it again,
        # so each wait-for-motor loop runs exactly one iteration
        async def start_motor(*args):
            _metrics(tracker).motor_stopped.clear()

        async def stop_motor(camera):
            _metrics(tracker).motor_stopped.set()

        for name in (
            "_move_continuous_timed",
            "_zoom_continuous_timed",
            "_zoom_absolute",
            "_move_relative",
        ):
            setattr(tracker.onvif, name, AsyncMock(side_effect=start_motor))
        tracker.onvif.get_camera_status = AsyncMock(side_effect=stop_motor)
        return tracker

    def test_pan_is_converted_to_timed_move(self) -> None:
        tracker = self._tracker()

        self._run_queue(tracker, (1.0, 0.5, 0.0, 0.0))

        tracker.onvif._move_continuous_timed.assert_awaited_once()
        camera, pan, tilt, duration = (
            tracker.onvif._move_continuous_timed.await_args.args
        )
        self.assertEqual(camera, CAMERA)
        self.assertAlmostEqual(pan, 0.3)
        self.assertAlmostEqual(tilt, 0.0)
        self.assertAlmostEqual(duration, 0.5 / (2.0 * 0.3))
        tracker.onvif._zoom_continuous_timed.assert_not_awaited()
        tracker.onvif.get_camera_status.assert_awaited()
        self.assertTrue(_metrics(tracker).motor_stopped.is_set())

    def test_relative_fov_with_relative_zoom_moves_in_one_call(self) -> None:
        tracker = self._tracker(mode="relative_fov", zooming=ZoomingModeEnum.relative)

        self._run_queue(tracker, (1.0, 0.5, -0.1, 0.2))

        tracker.onvif._move_relative.assert_awaited_once_with(CAMERA, 0.5, -0.1, 0.2, 1)

    def test_relative_fov_with_absolute_zoom_moves_then_zooms(self) -> None:
        tracker = self._tracker(mode="relative_fov", zooming=ZoomingModeEnum.absolute)
        _metrics(tracker).zoom_level.value = 0.2

        self._run_queue(tracker, (1.0, 0.5, 0.0, 0.6))

        tracker.onvif._move_relative.assert_awaited_once_with(CAMERA, 0.5, 0.0, 0, 1)
        tracker.onvif._zoom_absolute.assert_awaited_once_with(CAMERA, 0.6, 1)

    def test_tiny_pan_is_dropped(self) -> None:
        tracker = self._tracker()

        self._run_queue(tracker, (1.0, 0.01, 0.0, 0.0))

        tracker.onvif._move_continuous_timed.assert_not_awaited()

    def test_continuous_zoom_updates_estimated_position(self) -> None:
        tracker = self._tracker()
        tracker.estimated_zoom_position[CAMERA] = 0.5
        tracker.native_zoom_range[CAMERA] = (1.0, 30.0)

        self._run_queue(tracker, (1.0, 0.0, 0.0, 0.3))

        tracker.onvif._zoom_continuous_timed.assert_awaited_once()
        camera, velocity, duration = (
            tracker.onvif._zoom_continuous_timed.await_args.args
        )
        self.assertAlmostEqual(velocity, 0.3)
        self.assertAlmostEqual(duration, 1.0)
        self.assertAlmostEqual(tracker.estimated_zoom_position[CAMERA], 0.8)

    def test_estimated_position_is_clamped(self) -> None:
        tracker = self._tracker()
        tracker.estimated_zoom_position[CAMERA] = 0.05

        self._run_queue(tracker, (1.0, 0.0, 0.0, -0.3))

        self.assertEqual(tracker.estimated_zoom_position[CAMERA], 0.0)

    def test_sunba_without_relative_zoom_uses_timed_zoom(self) -> None:
        tracker = self._tracker(sunba=True, features=("pt", "zoom"))

        self._run_queue(tracker, (1.0, 0.0, 0.0, 0.3))

        tracker.onvif._zoom_continuous_timed.assert_awaited_once()

    def test_continuous_zoom_needs_zoom_feature(self) -> None:
        tracker = self._tracker(features=("pt",))

        self._run_queue(tracker, (1.0, 0.0, 0.0, 0.3))

        tracker.onvif._zoom_continuous_timed.assert_not_awaited()

    def test_absolute_zoom_in_continuous_mode(self) -> None:
        tracker = self._tracker(
            zooming=ZoomingModeEnum.absolute, features=("pt", "zoom-a")
        )
        _metrics(tracker).zoom_level.value = 0.2

        self._run_queue(tracker, (1.0, 0.0, 0.0, 0.6))

        tracker.onvif._zoom_absolute.assert_awaited_once_with(CAMERA, 0.6, 1)

    def test_relative_fov_with_continuous_zoom(self) -> None:
        tracker = self._tracker(mode="relative_fov")
        tracker.estimated_zoom_position[CAMERA] = 0.5

        self._run_queue(tracker, (1.0, 0.0, 0.0, -0.3))

        tracker.onvif._move_relative.assert_not_awaited()
        tracker.onvif._zoom_continuous_timed.assert_awaited_once()
        self.assertAlmostEqual(tracker.estimated_zoom_position[CAMERA], 0.2)

    def test_queue_is_drained_on_exit(self) -> None:
        tracker = self._tracker()

        self._run_queue(tracker, (1.0, 0.5, 0.0, 0.0))
        tracker.move_queues[CAMERA].put_nowait((2.0, 0.5, 0.0, 0.0))
        tracker.stop_event.is_set.side_effect = [True]
        asyncio.run(tracker._process_move_queue(CAMERA))

        self.assertTrue(tracker.move_queues[CAMERA].empty())
        tracker.onvif._move_continuous_timed.assert_awaited_once()


class TestConstructorState(unittest.TestCase):
    def test_initializes_continuous_mode_tables(self) -> None:
        config = MagicMock()
        config.cameras = {}

        with patch("frigate.ptz.autotrack.CameraConfigUpdateSubscriber"):
            tracker = PtzAutoTracker(config, MagicMock(), {}, MagicMock(), MagicMock())

        for name in (
            "movement_mode",
            "continuous_speed",
            "continuous_zoom_speed",
            "estimated_zoom_position",
            "native_zoom_range",
        ):
            self.assertEqual(getattr(tracker, name), {})
