"""Tests for the state topics MQTT publishes when it connects."""

import unittest
from unittest.mock import MagicMock

from frigate.comms.mqtt import MqttClient


def _published_state(camera: MagicMock, topic: str) -> object:
    config = MagicMock()
    config.cameras = {"ptz_cam": camera}
    config.mqtt.topic_prefix = "frigate"

    mqtt_client = MqttClient(config)
    mqtt_client.client = MagicMock()
    mqtt_client.connected = True
    mqtt_client._set_initial_topics()

    payloads = {
        call.args[0]: call.args[1] for call in mqtt_client.client.publish.call_args_list
    }
    return payloads[f"frigate/ptz_cam/{topic}"]


def _camera(*, autotracking: bool, autotracking_in_config: bool) -> MagicMock:
    camera = MagicMock()
    camera.motion.mask = {}
    camera.objects.mask = {}
    camera.zones = {}
    camera.onvif.autotracking.enabled = autotracking
    camera.onvif.autotracking.enabled_in_config = autotracking_in_config
    return camera


class TestMqttInitialAutotrackerState(unittest.TestCase):
    def test_reports_autotracking_disabled_at_runtime(self) -> None:
        # setup turns autotracking off when the PTZ is unreachable at startup,
        # and its OFF is lost if MQTT connects later or reconnects
        camera = _camera(autotracking=False, autotracking_in_config=True)

        self.assertEqual(_published_state(camera, "ptz_autotracker/state"), "OFF")

    def test_reports_autotracking_enabled(self) -> None:
        camera = _camera(autotracking=True, autotracking_in_config=True)

        self.assertEqual(_published_state(camera, "ptz_autotracker/state"), "ON")


if __name__ == "__main__":
    unittest.main()
