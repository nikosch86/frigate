"""Tests that WebPush config update polling is safe across threads."""

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from frigate.comms.webpush import WebPushClient


class TestWebPushConfigUpdates(unittest.TestCase):
    def _make_client(self) -> WebPushClient:
        config = MagicMock()
        config.cameras = {}
        stop_event = threading.Event()
        self.addCleanup(stop_event.set)

        with (
            patch("frigate.comms.webpush.Vapid01"),
            patch("frigate.comms.webpush.User") as user,
            patch("frigate.comms.webpush.ConfigSubscriber"),
            patch("frigate.comms.webpush.CameraConfigUpdateSubscriber"),
        ):
            user.select.return_value.dicts.return_value.iterator.return_value = []
            return WebPushClient(config, stop_event)

    def test_concurrent_publishes_poll_config_one_at_a_time(self):
        """publish runs on the mqtt, ipc, websocket and api threads, but the
        zmq sockets it polls are not thread safe: a concurrent drain can take
        another thread's second frame and leave it blocked in recv forever."""
        client = self._make_client()
        active = 0
        max_active = 0
        counter_lock = threading.Lock()

        def check_for_update():
            nonlocal active, max_active
            with counter_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            with counter_lock:
                active -= 1
            return (None, None)

        client.global_config_subscriber.check_for_update.side_effect = check_for_update
        client.config_subscriber.check_for_updates.return_value = {}

        def publish_stats():
            for _ in range(5):
                client.publish("stats", "{}")

        threads = [threading.Thread(target=publish_stats) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)

        self.assertEqual(max_active, 1)


if __name__ == "__main__":
    unittest.main()
