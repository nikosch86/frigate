"""Tests for autotracker state that must survive runtime config changes.

Regression coverage for a family of bugs where per-camera autotracker state was
built once at startup and never revisited. A camera that is added or enabled
after startup, or has autotracking enabled from the UI, would either raise a
KeyError on the autotracker thread or silently keep the wrong state:

- autotracker_init only got an entry for cameras enabled when PtzAutoTracker was
  constructed, so runtime-enabled cameras raised KeyError on lookup.
- ptz_metrics autotracker_enabled is what the camera processes read, but nothing
  updated it when autotracking was enabled through a config save, so it stayed
  False and the tracker never built a motion estimator.
- autotracking disabled at startup because a PTZ was unreachable stayed off
  after the camera came back, until a restart or a manual toggle.
"""

import asyncio
import threading
import time
import unittest
from unittest.mock import AsyncMock, MagicMock

from frigate.camera import PTZMetrics
from frigate.config import FrigateConfig
from frigate.ptz.autotrack import (
    OnvifSetupRetry,
    PtzAutoTracker,
    PtzAutoTrackerThread,
)

CAMERA = "ptz_cam"


def _config(autotracking_enabled: bool) -> FrigateConfig:
    return FrigateConfig(
        **{
            "mqtt": {"enabled": False},
            "cameras": {
                CAMERA: {
                    "ffmpeg": {
                        "inputs": [
                            {"path": "rtsp://10.0.0.1:554/video", "roles": ["detect"]}
                        ]
                    },
                    "detect": {"width": 1920, "height": 1080},
                    "zones": {"zone": {"coordinates": "0,0,1,0,1,1,0,1"}},
                    "onvif": {
                        "host": "10.0.0.1",
                        "autotracking": {
                            "enabled": autotracking_enabled,
                            "required_zones": ["zone"],
                        },
                    },
                }
            },
        }
    )


def _make_tracker(autotracking_enabled: bool = True) -> PtzAutoTracker:
    """Build a PtzAutoTracker without invoking __init__, which would try to set up
    onvif over the network. Only the config/metrics state is relevant here."""
    tracker = PtzAutoTracker.__new__(PtzAutoTracker)
    tracker.config = _config(autotracking_enabled)
    tracker.ptz_metrics = {CAMERA: PTZMetrics(autotracker_enabled=False)}
    tracker.onvif = MagicMock()
    tracker.config_subscriber = MagicMock()
    tracker.dispatcher = MagicMock()
    tracker.autotracker_init = {}
    tracker.calibrating = {}
    tracker.tracked_object = {}
    tracker.onvif_setup_retries = {}
    return tracker


class TestAutotrackerInitGuards(unittest.IsolatedAsyncioTestCase):
    async def test_camera_maintenance_returns_early_when_not_initialized(self) -> None:
        # a camera enabled at runtime has no autotracker_init entry, which used to
        # raise KeyError and kill the autotracker thread for every camera
        tracker = _make_tracker()
        self.assertNotIn(CAMERA, tracker.autotracker_init)

        await tracker.camera_maintenance(CAMERA)

        tracker.onvif.get_camera_status.assert_not_called()

    async def test_camera_maintenance_returns_early_when_init_incomplete(self) -> None:
        # autotracker_init is seeded False for enabled cameras before setup runs
        tracker = _make_tracker()
        tracker.autotracker_init[CAMERA] = False

        await tracker.camera_maintenance(CAMERA)

        tracker.onvif.get_camera_status.assert_not_called()


class TestAutotrackerMetricSync(unittest.TestCase):
    def test_metric_follows_config_when_enabled_by_update(self) -> None:
        # autotracking enabled via a config save: the metric was seeded False when
        # the camera was added and nothing else updates it
        tracker = _make_tracker(autotracking_enabled=True)
        metrics = tracker.ptz_metrics[CAMERA]
        self.assertFalse(metrics.autotracker_enabled.value)

        tracker.config_subscriber.check_for_updates.return_value = {"onvif": [CAMERA]}
        tracker.check_for_updates()

        self.assertTrue(metrics.autotracker_enabled.value)

    def test_metric_follows_config_when_disabled_by_update(self) -> None:
        tracker = _make_tracker(autotracking_enabled=False)
        metrics = tracker.ptz_metrics[CAMERA]
        metrics.autotracker_enabled.value = True

        tracker.config_subscriber.check_for_updates.return_value = {
            "autotracking": [CAMERA]
        }
        tracker.check_for_updates()

        self.assertFalse(metrics.autotracker_enabled.value)

    def test_metric_sync_skips_camera_without_metrics(self) -> None:
        # `add` reaches the maintainer and the autotracker on separate threads with
        # no ordering guarantee, so the metrics may not exist yet
        tracker = _make_tracker()
        tracker.ptz_metrics = {}
        tracker.config_subscriber.check_for_updates.return_value = {"add": [CAMERA]}

        tracker.check_for_updates()

    def test_metric_sync_skips_unknown_camera(self) -> None:
        tracker = _make_tracker()
        tracker.config_subscriber.check_for_updates.return_value = {
            "add": ["not_in_config"]
        }

        tracker.check_for_updates()


class TestAutotrackerOnvifRetry(unittest.TestCase):
    """Autotracking that setup disabled because ONVIF failed comes back on its own."""

    def setUp(self) -> None:
        # the probe runs on the ONVIF loop thread in production, so give it one
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.loop_thread.start()

        # as setup leaves it: configured on, switched off at runtime
        self.tracker = _make_tracker(autotracking_enabled=True)
        self.tracker.config.cameras[CAMERA].onvif.autotracking.enabled = False
        self.tracker.onvif.loop = self.loop
        self.tracker.autotracker_init[CAMERA] = False
        self.tracker.ptz_metrics[CAMERA].start_time.value = 5.0
        self.retry = OnvifSetupRetry(delay=30, next_attempt=0)
        self.tracker.onvif_setup_retries[CAMERA] = self.retry

    def tearDown(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=5)
        self.loop.close()

    def _probe_returns(self, **kwargs) -> AsyncMock:
        self.tracker.onvif.get_camera_info = AsyncMock(**kwargs)
        return self.tracker.onvif.get_camera_info

    def _finish_attempt(self) -> None:
        """Run one retry tick to start the probe and another once it finished."""
        self.tracker.retry_onvif_setup()
        self.retry.attempt.exception(timeout=5)
        self.tracker.retry_onvif_setup()

    def _autotracking(self) -> bool:
        return self.tracker.config.cameras[CAMERA].onvif.autotracking.enabled

    def test_reenables_autotracking_once_onvif_answers(self) -> None:
        probe = self._probe_returns(return_value={"name": CAMERA, "features": ["pt"]})

        self._finish_attempt()

        probe.assert_awaited_once_with(CAMERA)
        self.assertTrue(self._autotracking())
        metrics = self.tracker.ptz_metrics[CAMERA]
        self.assertTrue(metrics.autotracker_enabled.value)
        self.assertEqual(metrics.start_time.value, 0)
        self.tracker.dispatcher.publish.assert_called_once_with(
            f"{CAMERA}/ptz_autotracker/state", "ON", retain=True
        )
        self.assertNotIn(CAMERA, self.tracker.onvif_setup_retries)
        # the restore leaves setup to the next tracked object, as after a manual ON
        self.assertFalse(self.tracker.autotracker_init[CAMERA])

    def test_backs_off_while_onvif_is_unreachable(self) -> None:
        self._probe_returns(return_value={})

        self._finish_attempt()

        self.assertFalse(self._autotracking())
        self.assertFalse(self.tracker.ptz_metrics[CAMERA].autotracker_enabled.value)
        self.tracker.dispatcher.publish.assert_not_called()
        self.assertIs(self.tracker.onvif_setup_retries[CAMERA], self.retry)
        self.assertIsNone(self.retry.attempt)
        self.assertEqual(self.retry.delay, 60)
        self.assertAlmostEqual(self.retry.next_attempt - time.monotonic(), 60, delta=1)

    def test_probe_error_counts_as_unreachable(self) -> None:
        self._probe_returns(side_effect=OSError("no route to host"))

        self._finish_attempt()

        self.assertFalse(self._autotracking())
        self.assertEqual(self.retry.delay, 60)

    def test_backoff_is_capped(self) -> None:
        self._probe_returns(return_value={})
        self.retry.delay = 240

        self._finish_attempt()

        # the plan caps the backoff at 5 minutes
        self.assertEqual(self.retry.delay, 300)

    def test_waits_until_the_next_attempt_is_due(self) -> None:
        probe = self._probe_returns(return_value={"name": CAMERA})
        self.retry.next_attempt = time.monotonic() + 100

        self.tracker.retry_onvif_setup()

        probe.assert_not_called()
        self.assertIsNone(self.retry.attempt)

    def test_does_not_block_while_the_probe_runs(self) -> None:
        # an unreachable host can take seconds to time out, and the autotracker
        # thread also maintains the other cameras
        release = threading.Event()
        # a failed assertion would otherwise leave the executor thread blocked
        self.addCleanup(release.set)

        async def slow_probe(camera):
            await asyncio.get_running_loop().run_in_executor(None, release.wait)
            return {"name": camera}

        probe = self._probe_returns(side_effect=slow_probe)

        self.tracker.retry_onvif_setup()
        self.tracker.retry_onvif_setup()

        self.assertEqual(probe.call_count, 1)
        self.assertFalse(self._autotracking())

        release.set()
        self.retry.attempt.result(timeout=5)
        self.tracker.retry_onvif_setup()

        self.assertTrue(self._autotracking())

    def test_user_switching_on_keeps_the_retry(self) -> None:
        # the setup that follows a manual ON can fail and schedule a retry before
        # this thread sees the ON
        self.tracker.config.cameras[CAMERA].onvif.autotracking.enabled = True
        self.tracker.config_subscriber.check_for_updates.return_value = {
            "autotracking": [CAMERA]
        }

        self.tracker.check_for_updates()

        self.assertIs(self.tracker.onvif_setup_retries[CAMERA], self.retry)

    def test_user_change_cancels_the_retry(self) -> None:
        # a user switching autotracking off must not be overridden later
        probe = self._probe_returns(return_value={"name": CAMERA})
        self.tracker.config_subscriber.check_for_updates.return_value = {
            "autotracking": [CAMERA]
        }

        self.tracker.check_for_updates()
        self.tracker.retry_onvif_setup()

        probe.assert_not_called()
        self.assertFalse(self._autotracking())
        self.assertNotIn(CAMERA, self.tracker.onvif_setup_retries)

    def test_user_change_during_a_probe_wins(self) -> None:
        self._probe_returns(return_value={"name": CAMERA})
        self.tracker.retry_onvif_setup()
        self.retry.attempt.result(timeout=5)

        self.tracker.config_subscriber.check_for_updates.return_value = {
            "autotracking": [CAMERA]
        }
        self.tracker.check_for_updates()
        self.tracker.retry_onvif_setup()

        self.assertFalse(self._autotracking())
        self.tracker.dispatcher.publish.assert_not_called()

    def test_onvif_config_change_keeps_the_retry(self) -> None:
        # new ONVIF settings re-initialize the camera, which may now connect
        self.tracker.config_subscriber.check_for_updates.return_value = {
            "onvif": [CAMERA]
        }

        self.tracker.check_for_updates()

        self.assertIs(self.tracker.onvif_setup_retries[CAMERA], self.retry)

    def test_config_without_autotracking_drops_the_retry(self) -> None:
        # a UI save of the ONVIF section arrives as an onvif update, not an
        # autotracking one
        probe = self._probe_returns(return_value={"name": CAMERA})
        self.tracker.config.cameras[CAMERA].onvif.autotracking.enabled_in_config = False
        self.tracker.config_subscriber.check_for_updates.return_value = {
            "onvif": [CAMERA]
        }

        self.tracker.check_for_updates()
        self.tracker.retry_onvif_setup()

        probe.assert_not_called()
        self.assertFalse(self._autotracking())
        self.tracker.dispatcher.publish.assert_not_called()
        self.assertNotIn(CAMERA, self.tracker.onvif_setup_retries)

    def test_removed_camera_drops_the_retry(self) -> None:
        probe = self._probe_returns(return_value={})
        del self.tracker.config.cameras[CAMERA]

        self.tracker.retry_onvif_setup()

        probe.assert_not_called()
        self.assertNotIn(CAMERA, self.tracker.onvif_setup_retries)


class TestAutotrackerThreadTick(unittest.TestCase):
    def test_tick_applies_updates_before_retrying_onvif(self) -> None:
        # a user OFF applied earlier in the same tick must win over a restore
        thread = PtzAutoTrackerThread.__new__(PtzAutoTrackerThread)
        thread.config = MagicMock()
        thread.config.cameras = {}
        thread.stop_event = MagicMock()
        thread.stop_event.wait.side_effect = [False, True]
        thread.ptz_autotracker = MagicMock()

        thread.run()

        calls = [name for name, _, _ in thread.ptz_autotracker.method_calls]
        self.assertEqual(calls[:2], ["check_for_updates", "retry_onvif_setup"])


if __name__ == "__main__":
    unittest.main()
