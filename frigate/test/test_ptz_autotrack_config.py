"""Tests for the continuous-zoom and Sunba fields of the ONVIF autotracking config."""

import unittest

from pydantic import ValidationError

from frigate.config.camera.onvif import (
    OnvifConfig,
    PtzAutotrackConfig,
    ZoomingModeEnum,
)


class TestZoomingModeEnum(unittest.TestCase):
    def test_continuous_mode_is_available(self) -> None:
        self.assertEqual(ZoomingModeEnum("continuous"), ZoomingModeEnum.continuous)
        self.assertEqual(
            PtzAutotrackConfig(zooming="continuous").zooming,
            ZoomingModeEnum.continuous,
        )


class TestContinuousSpeedFields(unittest.TestCase):
    def test_defaults(self) -> None:
        config = PtzAutotrackConfig()

        self.assertEqual(config.continuous_speed, 2.0)
        self.assertEqual(config.continuous_zoom_speed, 1.0)
        self.assertIsNone(config.assumed_zoom_range)
        self.assertIsNone(config.preset_zoom_level)

    def test_continuous_speed_accepts_bounds_inclusive(self) -> None:
        self.assertEqual(PtzAutotrackConfig(continuous_speed=0.1).continuous_speed, 0.1)
        self.assertEqual(
            PtzAutotrackConfig(continuous_speed=10.0).continuous_speed, 10.0
        )

    def test_continuous_speed_rejects_out_of_bounds(self) -> None:
        with self.assertRaises(ValidationError):
            PtzAutotrackConfig(continuous_speed=0.05)
        with self.assertRaises(ValidationError):
            PtzAutotrackConfig(continuous_speed=10.5)

    def test_continuous_zoom_speed_accepts_bounds_inclusive(self) -> None:
        self.assertEqual(
            PtzAutotrackConfig(continuous_zoom_speed=0.1).continuous_zoom_speed, 0.1
        )
        self.assertEqual(
            PtzAutotrackConfig(continuous_zoom_speed=5.0).continuous_zoom_speed, 5.0
        )

    def test_continuous_zoom_speed_rejects_out_of_bounds(self) -> None:
        with self.assertRaises(ValidationError):
            PtzAutotrackConfig(continuous_zoom_speed=0.0)
        with self.assertRaises(ValidationError):
            PtzAutotrackConfig(continuous_zoom_speed=5.1)


class TestAssumedZoomRange(unittest.TestCase):
    def test_none_is_kept(self) -> None:
        self.assertIsNone(
            PtzAutotrackConfig(assumed_zoom_range=None).assumed_zoom_range
        )

    def test_int_list_is_coerced_to_floats(self) -> None:
        config = PtzAutotrackConfig(assumed_zoom_range=[1, 30])

        self.assertEqual(config.assumed_zoom_range, [1.0, 30.0])
        self.assertTrue(all(isinstance(v, float) for v in config.assumed_zoom_range))

    def test_tuple_is_accepted(self) -> None:
        config = PtzAutotrackConfig(assumed_zoom_range=(2.5, 12))

        self.assertEqual(config.assumed_zoom_range, [2.5, 12.0])

    def test_rejects_min_not_below_max(self) -> None:
        with self.assertRaisesRegex(ValidationError, "min < max"):
            PtzAutotrackConfig(assumed_zoom_range=[30, 1])
        with self.assertRaisesRegex(ValidationError, "min < max"):
            PtzAutotrackConfig(assumed_zoom_range=[5, 5])

    def test_rejects_non_positive_min(self) -> None:
        with self.assertRaisesRegex(ValidationError, "must be positive"):
            PtzAutotrackConfig(assumed_zoom_range=[0, 30])
        with self.assertRaisesRegex(ValidationError, "must be positive"):
            PtzAutotrackConfig(assumed_zoom_range=[-1, 30])

    def test_rejects_wrong_length(self) -> None:
        with self.assertRaisesRegex(ValidationError, "list of two numbers"):
            PtzAutotrackConfig(assumed_zoom_range=[1])
        with self.assertRaisesRegex(ValidationError, "list of two numbers"):
            PtzAutotrackConfig(assumed_zoom_range=[1, 10, 30])

    def test_rejects_non_numeric_entries(self) -> None:
        with self.assertRaisesRegex(ValidationError, "list of two numbers"):
            PtzAutotrackConfig(assumed_zoom_range=["1", "30"])

    def test_rejects_non_sequence(self) -> None:
        with self.assertRaisesRegex(ValidationError, "list of two numbers"):
            PtzAutotrackConfig(assumed_zoom_range=30)


class TestPresetZoomLevel(unittest.TestCase):
    def test_preset_without_range_is_accepted(self) -> None:
        self.assertEqual(
            PtzAutotrackConfig(preset_zoom_level=10).preset_zoom_level, 10.0
        )

    def test_range_without_preset_is_accepted(self) -> None:
        config = PtzAutotrackConfig(assumed_zoom_range=[1, 30])

        self.assertIsNone(config.preset_zoom_level)

    def test_preset_inside_range_is_accepted(self) -> None:
        config = PtzAutotrackConfig(assumed_zoom_range=[1, 30], preset_zoom_level=10)

        self.assertEqual(config.preset_zoom_level, 10.0)

    def test_preset_at_range_bounds_is_accepted(self) -> None:
        low = PtzAutotrackConfig(assumed_zoom_range=[1, 30], preset_zoom_level=1)
        high = PtzAutotrackConfig(assumed_zoom_range=[1, 30], preset_zoom_level=30)

        self.assertEqual(low.preset_zoom_level, 1.0)
        self.assertEqual(high.preset_zoom_level, 30.0)

    def test_preset_outside_range_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationError, "must be within"):
            PtzAutotrackConfig(assumed_zoom_range=[1, 30], preset_zoom_level=31)
        with self.assertRaisesRegex(ValidationError, "must be within"):
            PtzAutotrackConfig(assumed_zoom_range=[2, 30], preset_zoom_level=1)


class TestOnvifConfigSunbaQuirks(unittest.TestCase):
    def test_defaults_to_disabled(self) -> None:
        self.assertFalse(OnvifConfig().sunba_quirks)

    def test_can_be_enabled(self) -> None:
        self.assertTrue(OnvifConfig(sunba_quirks=True).sunba_quirks)

    def test_autotracking_continuous_fields_nest_under_onvif(self) -> None:
        config = OnvifConfig(
            sunba_quirks=True,
            autotracking={
                "zooming": "continuous",
                "assumed_zoom_range": [1, 20],
                "preset_zoom_level": 4,
            },
        )

        self.assertEqual(config.autotracking.zooming, ZoomingModeEnum.continuous)
        self.assertEqual(config.autotracking.assumed_zoom_range, [1.0, 20.0])
        self.assertEqual(config.autotracking.preset_zoom_level, 4.0)
