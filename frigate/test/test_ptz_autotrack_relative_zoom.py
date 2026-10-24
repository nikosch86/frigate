"""Tests for relative autotracking zoom on cameras that scale it to the zoom range."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from frigate.config import ZoomingModeEnum
from frigate.config.camera.onvif import RelativeZoomScaleEnum
from frigate.const import AUTOTRACKING_MAX_AREA_RATIO
from frigate.ptz.autotrack import PtzAutoTracker
from frigate.test.test_ptz_autotrack_continuous import (
    CAMERA,
    _make_tracker,
    _metrics,
    _obj,
)

# a 1x-4x camera whose zoom position and relative zoom are linear in magnification
ZOOM_RANGE = (1.0, 4.0)
FRAME = (720, 1280)
FRAME_AREA = FRAME[0] * FRAME[1]
# the zoom factor of the camera this scale was built for, and the default
ZOOM_FACTORS = (0.5, 0.3)
FRAME_TIME = 1.79e9


def _limit(zoom_factor: float) -> float:
    return AUTOTRACKING_MAX_AREA_RATIO ** (1 / zoom_factor)


def _limit_side(zoom_factor: float) -> float:
    """Return the side of a square object whose target box is the limit."""
    return (_limit(zoom_factor) ** (1 / zoom_factor) * FRAME_AREA) ** 0.5


def _box(side: float) -> list[float]:
    return [640 - side / 2, 360 - side / 2, 640 + side / 2, 360 + side / 2]


def _tracker(
    decision=None,
    level: float = 0.0,
    scale: RelativeZoomScaleEnum = RelativeZoomScaleEnum.range,
    zoom_factor: float = ZOOM_FACTORS[0],
    zoom_range: tuple[float, float] = ZOOM_RANGE,
    frame_shape: tuple[int, int] = FRAME,
    **overrides,
) -> PtzAutoTracker:
    tracker = _make_tracker(
        features=("pt", "pt-r-fov", "zoom", "zoom-r", "zoom-a"),
        zooming=ZoomingModeEnum.relative,
        relative_zoom_scale=scale,
        assumed_zoom_range=list(zoom_range),
        frame_shape=frame_shape,
        fps=5,
        **overrides,
    )
    tracker.zoom_factor[CAMERA] = zoom_factor
    tracker.tracked_object_metrics[CAMERA]["max_target_box"] = _limit(zoom_factor)
    tracker._should_zoom_in = MagicMock(return_value=decision)
    _metrics(tracker).zoom_level.value = level
    return tracker


def _magnification(level: float, zoom_range=ZOOM_RANGE) -> float:
    return zoom_range[0] + level * (zoom_range[1] - zoom_range[0])


def _target_box_after(
    target_box: float, level: float, zoom: float, zoom_factor: float
) -> float:
    # target_box is the object's area to the power of zoom_factor, and the area
    # grows with the square of the magnification
    new_level = min(1.0, max(0.0, level + zoom))
    scale = _magnification(new_level) / _magnification(level)
    return target_box * scale ** (2 * zoom_factor)


def _target_box_for(
    magnification_ratio: float, zoom_factor: float = ZOOM_FACTORS[0]
) -> float:
    """Return the target box that needs this change of magnification."""
    return _limit(zoom_factor) / magnification_ratio ** (2 * zoom_factor)


class TestGetZoomAmountRangeScale(unittest.TestCase):
    OBJ = _obj([600, 320, 680, 400])

    def _zoom(self, tracker: PtzAutoTracker) -> float:
        return tracker._get_zoom_amount(CAMERA, self.OBJ, self.OBJ.obj_data["box"], 0)

    def _zoom_for(self, decision, level: float, ratio: float, **kwargs) -> float:
        tracker = _tracker(decision, level, **kwargs)
        zoom_factor = kwargs.get("zoom_factor", ZOOM_FACTORS[0])
        tracker.tracked_object_metrics[CAMERA]["target_box"] = _target_box_for(
            ratio, zoom_factor
        )
        return self._zoom(tracker)

    def _assert_reaches_limit(self, decision: bool, level: float, ratio: float):
        for zoom_factor in ZOOM_FACTORS:
            with self.subTest(level=level, ratio=ratio, zoom_factor=zoom_factor):
                zoom = self._zoom_for(decision, level, ratio, zoom_factor=zoom_factor)

                self.assertEqual(zoom > 0, decision)
                self.assertAlmostEqual(
                    _target_box_after(
                        _target_box_for(ratio, zoom_factor), level, zoom, zoom_factor
                    ),
                    _limit(zoom_factor),
                )

    def test_zoom_in_brings_the_object_to_the_limit(self) -> None:
        for level, ratio in ((0.0, 2.0), (1 / 3, 1.1), (0.5, 1.4)):
            self._assert_reaches_limit(True, level, ratio)

    def test_zoom_out_brings_the_object_to_the_limit(self) -> None:
        for level, ratio in ((1.0, 0.8), (0.5, 0.5), (2 / 3, 0.7)):
            self._assert_reaches_limit(False, level, ratio)

    def test_small_correction_is_a_small_zoom(self) -> None:
        # 2x to 2.2x is a fifteenth of the 1x-4x range, where the frame scale
        # sends nearly the whole range
        self.assertAlmostEqual(self._zoom_for(True, 1 / 3, 1.1), 0.2 / 3)

    def test_zoom_stops_at_the_end_of_the_range(self) -> None:
        self.assertAlmostEqual(self._zoom_for(True, 2 / 3, 2.0), 1 / 3)
        self.assertAlmostEqual(self._zoom_for(False, 1 / 3, 0.25), -1 / 3)

    def test_zoom_range_does_not_have_to_start_at_1x(self) -> None:
        # 5x to 6x on a 2x-8x camera is a sixth of its range
        self.assertAlmostEqual(
            self._zoom_for(True, 0.5, 1.2, zoom_range=(2.0, 8.0)), 1 / 6
        )

    def test_zoom_out_for_edges_or_speed_takes_at_least_a_tenth(self) -> None:
        # a small object, one at the limit, one just over it, and a small object
        # at max zoom, where the range has nothing left to zoom in to
        for level, ratio in ((0.5, 2.0), (0.5, 1.0), (0.5, 0.98), (1.0, 2.0)):
            with self.subTest(level=level, ratio=ratio):
                self.assertAlmostEqual(self._zoom_for(False, level, ratio), -0.1)

    def test_zoom_out_for_edges_or_speed_keeps_a_larger_zoom(self) -> None:
        # 4x to 3.68x is more than a tenth of the range
        self.assertAlmostEqual(self._zoom_for(False, 1.0, 0.92), -0.32 / 3)

    def test_no_decision_means_no_zoom(self) -> None:
        self.assertEqual(self._zoom_for(None, 0.5, 2.0), 0)

    def test_first_zoom_brings_a_new_object_to_the_limit(self) -> None:
        # an object needing 2x from 1x, and one needing 1.5x less from 4x
        for level, scale in ((0.0, 0.5), (1.0, 1.5)):
            for zoom_factor in ZOOM_FACTORS:
                with self.subTest(level=level, scale=scale, zoom_factor=zoom_factor):
                    tracker = _tracker(level=level, zoom_factor=zoom_factor)
                    side = _limit_side(zoom_factor) * scale
                    obj = _obj(_box(side))

                    zoom = tracker._get_zoom_amount(CAMERA, obj, obj.obj_data["box"], 0)

                    self.assertEqual(zoom > 0, scale < 1)
                    self.assertAlmostEqual(
                        _target_box_after(
                            (side * side / FRAME_AREA) ** zoom_factor,
                            level,
                            zoom,
                            zoom_factor,
                        ),
                        _limit(zoom_factor),
                    )

    def test_first_zoom_skips_a_box_without_area(self) -> None:
        tracker = _tracker(level=0.5)
        obj = _obj([600, 320, 600, 320])

        self.assertEqual(
            tracker._get_zoom_amount(CAMERA, obj, obj.obj_data["box"], 0), 0
        )

    def test_frame_scale_sends_one_minus_the_calculation(self) -> None:
        tracker = _tracker(True, 0.5, scale=RelativeZoomScaleEnum.frame)
        tracker.tracked_object_metrics[CAMERA]["target_box"] = _limit(0.5) / 1.5
        # (1.5 - 1) / (1.5 + 1) = 0.2, sent as 1 - 0.2
        self.assertAlmostEqual(self._zoom(tracker), 0.8)

        tracker = _tracker(False, 0.5, scale=RelativeZoomScaleEnum.frame)
        tracker.tracked_object_metrics[CAMERA]["target_box"] = _limit(0.5) * 2
        # (0.5 - 1) / (0.5 + 1) = -1/3, sent as -(2 * -1/3 + 1)
        self.assertAlmostEqual(self._zoom(tracker), -1 / 3)

    def test_frame_scale_first_zoom_sends_the_target_box(self) -> None:
        tracker = _tracker(
            level=0.5, scale=RelativeZoomScaleEnum.frame, zoom_factor=0.3
        )
        small = _obj([600, 320, 640, 360])
        # a 40px object is under the zoom limit, so the first zoom is its target
        # box, which zooms in
        self.assertAlmostEqual(
            tracker._get_zoom_amount(CAMERA, small, small.obj_data["box"], 0),
            (40 * 40 / FRAME_AREA) ** 0.3,
        )

        tracker = _tracker(
            level=0.5, scale=RelativeZoomScaleEnum.frame, zoom_factor=0.3
        )
        large = _obj([400, 200, 880, 520])
        # a 480px object is over the zoom limit, so the first zoom is sent as one
        # minus its target box, which zooms out
        self.assertAlmostEqual(
            tracker._get_zoom_amount(CAMERA, large, large.obj_data["box"], 0),
            -(1 - (480 * 480 / FRAME_AREA) ** 0.3),
        )


class TestRangeScaleSettles(unittest.TestCase):
    """A person standing still in the middle of a 1x-4x range scale camera.

    Each decision sees the size the person has after the previous zoom, through
    the real zoom decision, on the zoom-only path (no predicted move time) or the
    pan path (a predicted move time with the area regression fitted).
    """

    def _track(
        self,
        best: float,
        level: float,
        zoom_factor: float,
        predicted_times: tuple[float, ...] = (0,),
    ) -> tuple[list[bool], float]:
        """Zoom until no zoom is decided; return the directions and magnification."""
        tracker = _tracker(level=level, zoom_factor=zoom_factor)
        del tracker._should_zoom_in
        _metrics(tracker).min_zoom.value = 0.0
        _metrics(tracker).max_zoom.value = 1.0
        tracker.tracked_object_metrics[CAMERA]["velocity"] = np.zeros((4,))
        tracker.tracked_object_history[CAMERA].append({"frame_time": FRAME_TIME})
        side_at_1x = _limit_side(zoom_factor) / best
        directions = []

        for step in range(10):
            level = _metrics(tracker).zoom_level.value
            side = side_at_1x * _magnification(level)
            box = _box(side)
            tracker.tracked_object_metrics[CAMERA]["target_box"] = (
                side * side / FRAME_AREA
            ) ** zoom_factor
            # fitted through the origin on frame times, the regression predicts
            # about the whole area, as it does in production
            tracker.tracked_object_metrics[CAMERA]["area_coefficients"] = np.array(
                [side * side / FRAME_TIME]
            )

            zoom = tracker._get_zoom_amount(
                CAMERA, _obj(box), box, predicted_times[step % len(predicted_times)]
            )

            if zoom == 0:
                break

            directions.append(zoom > 0)
            _metrics(tracker).zoom_level.value = min(1.0, max(0.0, level + zoom))

        return directions, _magnification(_metrics(tracker).zoom_level.value)

    def _assert_settles(self, best, level, zoom_factors, zoom_in, **kwargs):
        for zoom_factor in zoom_factors:
            with self.subTest(best=best, zoom_factor=zoom_factor, **kwargs):
                directions, magnification = self._track(
                    best, level, zoom_factor, **kwargs
                )

                self.assertEqual(directions, [zoom_in])
                self.assertAlmostEqual(magnification, min(best, ZOOM_RANGE[1]))

    def test_zooms_in_once(self) -> None:
        for best in (1.2, 1.5, 2.0, 3.0):
            self._assert_settles(best, 0.0, ZOOM_FACTORS, True)

    def test_zooms_out_once(self) -> None:
        for best in (1.2, 2.0, 3.0):
            self._assert_settles(best, 1.0, ZOOM_FACTORS, False)

    def test_object_too_small_for_the_range_stays_at_max_zoom(self) -> None:
        self._assert_settles(6.0, 0.0, ZOOM_FACTORS, True)

    def test_pan_moves_do_not_zoom_back_out(self) -> None:
        # the area prediction only shifts the limit noticeably at higher zoom
        # factors, on the pan path that alternates with zoom-only moves
        for predicted_times in ((0, 0.5), (0.5, 0)):
            for best in (1.5, 2.0, 3.0):
                self._assert_settles(
                    best, 0.0, (0.5, 0.75), True, predicted_times=predicted_times
                )
            self._assert_settles(
                2.0, 1.0, (0.5,), False, predicted_times=predicted_times
            )


class TestAutotrackMovePtzZoomPrediction(unittest.TestCase):
    # intercept 1 with move coefficients (0.2, 1) predicts a 0.2s move plus 1s
    # per unit of pan/tilt distance, and a zoom time of 1s per unit of zoom
    WEIGHTS = ["0", "1", "1", "0.2", "1", "1"]

    def _pan(self, scale: RelativeZoomScaleEnum) -> float:
        tracker = _tracker(
            level=0.5,
            scale=scale,
            frame_shape=(1000, 1000),
            movement_weights=self.WEIGHTS,
        )
        tracker.movement_mode[CAMERA] = "relative_fov"
        tracker.intercept[CAMERA] = 1.0
        tracker.move_coefficients[CAMERA] = [0.2, 1.0]
        tracker.zoom_time[CAMERA] = 1.0
        tracker.tracked_object_metrics[CAMERA].update(
            {"velocity": np.array([10.0, 0.0, 10.0, 0.0]), "valid_velocity": True}
        )
        tracker._get_zoom_amount = MagicMock(return_value=0.2)
        tracker._enqueue_move = MagicMock()

        tracker._autotrack_move_ptz(CAMERA, _obj([600, 450, 700, 550]))

        tracker._enqueue_move.assert_called_once()
        _, _, pan, tilt, zoom = tracker._enqueue_move.call_args.args
        self.assertEqual(zoom, 0.2)
        self.assertAlmostEqual(tilt, 0.0)
        return pan

    def test_range_scale_leads_by_the_zoom_it_sends(self) -> None:
        # a 0.2 zoom takes 0.2s, which moves the box another 10px at 5 fps
        self.assertAlmostEqual(self._pan(RelativeZoomScaleEnum.range), 0.37)

    def test_frame_scale_leads_by_one_minus_the_zoom(self) -> None:
        # the frame scale takes a 0.2 zoom as a 0.8 change, clamped to the 0.5
        # left above level 0.5, so 0.5s and another 25px
        self.assertAlmostEqual(self._pan(RelativeZoomScaleEnum.frame), 0.4)


class TestCalibrationRelativeZoom(unittest.TestCase):
    def _calibration_zooms(self, scale: RelativeZoomScaleEnum) -> list[float]:
        tracker = _tracker(scale=scale)
        tracker.onvif.cams[CAMERA]["absolute_zoom_range"] = {
            "XRange": {"Min": 0.0, "Max": 1.0}
        }
        tracker._calculate_move_coefficients = MagicMock()

        async def release_motor(camera):
            _metrics(tracker).motor_stopped.set()

        tracker.onvif.get_camera_status = AsyncMock(side_effect=release_motor)
        tracker.onvif._zoom_absolute = AsyncMock(return_value=True)
        tracker.onvif._move_relative = AsyncMock(return_value=True)
        tracker.onvif._move_to_preset = AsyncMock(return_value=True)

        with self.assertLogs("frigate.ptz.autotrack", level="INFO"):
            asyncio.run(tracker._calibrate_camera(CAMERA))

        return [
            call.args[3]
            for call in tracker.onvif._move_relative.await_args_list
            if call.args[3] != 0
        ]

    def test_range_scale_zooms_across_the_whole_range(self) -> None:
        self.assertEqual(
            self._calibration_zooms(RelativeZoomScaleEnum.range), [-1, 1, -1, 1]
        )

    def test_frame_scale_keeps_the_small_zooms(self) -> None:
        self.assertEqual(
            self._calibration_zooms(RelativeZoomScaleEnum.frame),
            [-1e-2, 1e-2, -1e-2, 1e-2],
        )


if __name__ == "__main__":
    unittest.main()
