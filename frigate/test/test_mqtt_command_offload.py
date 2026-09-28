"""Tests that MQTT commands are handled off the paho network thread."""

import threading
import unittest
from unittest.mock import MagicMock, patch

from frigate.comms.mqtt import MqttClient


def _start_client(dispatcher) -> MqttClient:
    config = MagicMock()
    config.cameras = {}
    config.notifications.enabled_in_config = False
    config.mqtt.topic_prefix = "frigate"
    config.mqtt.client_id = "frigate"
    config.mqtt.user = None
    config.mqtt.tls_ca_certs = None
    config.mqtt.tls_insecure = None

    with patch("frigate.comms.mqtt.mqtt.Client"):
        mqtt_client = MqttClient(config)
        mqtt_client.subscribe(dispatcher)

    return mqtt_client


def _message(topic: str, payload: bytes) -> MagicMock:
    message = MagicMock()
    message.topic = topic
    message.payload = payload
    return message


class TestMqttCommandOffload(unittest.TestCase):
    def test_blocking_command_does_not_block_network_thread(self):
        """A dispatcher that never returns must not stall the paho callback,
        otherwise keepalives stop and the broker silently drops Frigate."""
        release = threading.Event()
        received = []
        handled = threading.Event()

        def dispatcher(topic, payload):
            release.wait()
            received.append((topic, payload))
            handled.set()

        mqtt_client = _start_client(dispatcher)
        callback_done = threading.Event()

        def paho_callback():
            mqtt_client.on_mqtt_command(
                None, None, _message("frigate/cam/detect/set", b"ON")
            )
            callback_done.set()

        threading.Thread(target=paho_callback, daemon=True).start()

        self.assertTrue(callback_done.wait(2))
        release.set()
        self.assertTrue(handled.wait(2))
        self.assertEqual(received, [("cam/detect/set", "ON")])

    def test_failing_command_does_not_stop_later_commands(self):
        received = []
        handled = threading.Event()

        def dispatcher(topic, payload):
            if payload == "boom":
                raise ValueError("boom")
            received.append((topic, payload))
            handled.set()

        mqtt_client = _start_client(dispatcher)

        mqtt_client.on_mqtt_command(None, None, _message("frigate/restart", b"boom"))
        mqtt_client.on_mqtt_command(None, None, _message("frigate/restart", b"\xff"))
        mqtt_client.on_mqtt_command(
            None, None, _message("frigate/cam/detect/set", b"OFF")
        )

        self.assertTrue(handled.wait(2))
        self.assertEqual(received, [("cam/detect/set", "OFF")])

    def test_stop_ends_command_thread(self):
        mqtt_client = _start_client(MagicMock())

        mqtt_client.stop()

        mqtt_client._command_thread.join(2)
        self.assertFalse(mqtt_client._command_thread.is_alive())


if __name__ == "__main__":
    unittest.main()
