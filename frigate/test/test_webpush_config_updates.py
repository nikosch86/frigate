"""Tests that WebPush config update polling is safe across threads."""

import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from frigate.comms.webpush import WebPushClient
from frigate.config.auth import AuthConfig


class TestWebPushConfigUpdates(unittest.TestCase):
    def _make_client(self, cameras: list[str] | None = None) -> WebPushClient:
        config = MagicMock()
        config.cameras = {name: SimpleNamespace(name=name) for name in cameras or []}
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

    def _stored_users(self, users: list[dict[str, str]]) -> MagicMock:
        """Serve these users from the database while keeping the real
        role-to-camera resolution."""
        patcher = patch("frigate.comms.webpush.User.select")
        select = patcher.start()
        self.addCleanup(patcher.stop)
        select.return_value.dicts.return_value.iterator.return_value = users
        return select

    def _global_updates(self, client: WebPushClient, *updates) -> None:
        client.global_config_subscriber.check_for_update.side_effect = [
            *updates,
            (None, None),
        ]
        client.config_subscriber.check_for_updates.return_value = {}

    def test_concurrent_publishes_poll_config_one_at_a_time(self):
        """publish runs on several threads but zmq sockets are not thread safe:
        a concurrent drain can take another thread's second frame and leave that
        thread blocked in recv until the next config update."""
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

    def test_notifications_update_replaces_notification_config(self):
        client = self._make_client()
        notifications = MagicMock()
        self._global_updates(client, ("config/notifications", notifications))

        client.publish("stats", "{}")

        self.assertIs(client.config.notifications, notifications)

    def test_empty_notifications_update_keeps_current_config(self):
        client = self._make_client()
        current = client.config.notifications
        self._global_updates(client, ("config/notifications", None))

        client.publish("stats", "{}")

        self.assertIs(client.config.notifications, current)

    def test_auth_update_applies_new_roles_to_camera_access(self):
        client = self._make_client(["front_door", "back_yard"])
        self._stored_users([{"username": "alice", "role": "front_only"}])
        auth = AuthConfig(roles={"front_only": ["front_door"]})
        self._global_updates(client, ("config/auth", auth))

        client.publish("stats", "{}")

        self.assertIs(client.config.auth, auth)
        self.assertEqual(client.user_cameras, {"alice": {"front_door"}})

    def test_auth_update_without_config_refreshes_users_from_database(self):
        """User create, delete and role changes publish config/auth with no
        payload, so camera access must be reloaded from the database."""
        client = self._make_client(["front_door", "back_yard"])
        auth = AuthConfig(roles={"front_only": ["front_door"]})
        client.config.auth = auth
        self._stored_users(
            [
                {"username": "alice", "role": "front_only"},
                {"username": "bob", "role": "viewer"},
            ]
        )
        self._global_updates(client, ("config/auth", None))

        client.publish("stats", "{}")

        self.assertIs(client.config.auth, auth)
        self.assertEqual(
            client.user_cameras,
            {"alice": {"front_door"}, "bob": {"front_door", "back_yard"}},
        )

    def test_added_camera_is_tracked_and_granted_to_users(self):
        client = self._make_client(["front_door"])
        client.config.auth = AuthConfig()
        client.suspended_cameras["front_door"] = 123
        self._stored_users([{"username": "bob", "role": "viewer"}])
        client.global_config_subscriber.check_for_update.return_value = (None, None)

        def check_for_updates():
            # the real subscriber adds the new camera to the shared config
            client.config.cameras["garage"] = SimpleNamespace(name="garage")
            return {"add": ["garage"]}

        client.config_subscriber.check_for_updates.side_effect = check_for_updates

        client.publish("stats", "{}")

        self.assertEqual(client.suspended_cameras, {"front_door": 123, "garage": 0})
        self.assertEqual(client.last_camera_notification_time["garage"], 0)
        self.assertEqual(client.user_cameras, {"bob": {"front_door", "garage"}})


if __name__ == "__main__":
    unittest.main()
