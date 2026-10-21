"""Tests for re-initializing a camera's ONVIF client after a config save.

The autotracker can be polling the camera for its motor to stop while the UI saves
its ONVIF settings. A status call on a closed session fails without yielding, so a
poll against the old session spun forever and froze the whole ONVIF loop.
"""

import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from frigate.camera import PTZMetrics
from frigate.ptz.onvif import OnvifController

CAMERA = "ptz"


class TestReinitWhileTheAutotrackerWaits(unittest.TestCase):
    def setUp(self) -> None:
        controller = OnvifController.__new__(OnvifController)
        controller.loop = asyncio.new_event_loop()
        controller.loop_thread = threading.Thread(
            target=controller.loop.run_forever, daemon=True
        )
        controller.loop_thread.start()
        self.addCleanup(self._stop_loop)
        self.controller = controller

        controller.config = MagicMock()
        controller.config.cameras = {CAMERA: MagicMock()}
        controller.camera_configs = {CAMERA: controller.config.cameras[CAMERA]}
        controller.failed_cams = {}
        controller.ptz_metrics = {CAMERA: PTZMetrics(autotracker_enabled=True)}
        controller.ptz_metrics[CAMERA].motor_stopped.clear()
        self._on_loop(self._create_lock())

        self.old_session_closed = False

        async def get_status(request):
            # a closed aiohttp session raises before awaiting anything
            if self.old_session_closed:
                raise RuntimeError("Session is closed")

            await asyncio.sleep(0.01)
            raise ConnectionError("still moving, status unknown")

        async def close_old_session():
            # the real ONVIFCamera.close() yields while it closes each session
            self.old_session_closed = True
            await asyncio.sleep(0.05)

        self.old_client = MagicMock(close=AsyncMock(side_effect=close_old_session))
        controller.cams = {
            CAMERA: {
                "onvif": self.old_client,
                "init": True,
                "ptz": SimpleNamespace(GetStatus=get_status),
                "status_request": MagicMock(),
            }
        }

        async def new_client_cannot_connect_yet(camera_name):
            await asyncio.sleep(0.01)
            return False

        controller._init_onvif = new_client_cannot_connect_yet

    def _stop_loop(self) -> None:
        loop = self.controller.loop
        loop.call_soon_threadsafe(loop.stop)
        self.controller.loop_thread.join(timeout=5)

        tasks = asyncio.all_tasks(loop)
        for task in tasks:
            task.cancel()

        # an empty gather would bind to another loop
        if tasks:
            loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))

        loop.close()

    def _on_loop(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.controller.loop).result(
            timeout=5
        )

    async def _create_lock(self) -> None:
        self.controller.status_locks = {CAMERA: asyncio.Lock()}

    async def _wait_for_motor(self) -> None:
        # the autotracker's wait after every move and before a preset return
        while not self.controller.ptz_metrics[CAMERA].motor_stopped.is_set():
            await self.controller.get_camera_status(CAMERA)

    def _reinit_while_waiting(self) -> None:
        asyncio.run_coroutine_threadsafe(self._wait_for_motor(), self.controller.loop)
        reinit = asyncio.run_coroutine_threadsafe(
            self.controller._reinit_camera(CAMERA), self.controller.loop
        )

        # a frozen loop never finishes the re-init
        reinit.result(timeout=5)

    def test_reinit_finishes_and_the_poll_moves_to_the_new_client(self) -> None:
        new_client = MagicMock(close=AsyncMock())

        async def install_new_client(cam_name):
            self.controller.cams[cam_name] = {"onvif": new_client, "init": False}
            return True

        self.controller._init_single_camera = install_new_client

        self._reinit_while_waiting()

        self.old_client.close.assert_awaited_once()
        self.assertIs(self.controller.cams[CAMERA]["onvif"], new_client)
        new_client.close.assert_not_awaited()

    def test_removing_onvif_releases_the_poll(self) -> None:
        self.controller.config.cameras[CAMERA].onvif.host = None

        self._reinit_while_waiting()

        self.old_client.close.assert_awaited_once()
        self.assertNotIn(CAMERA, self.controller.cams)
        self.assertTrue(self.controller.ptz_metrics[CAMERA].motor_stopped.is_set())

    def test_status_of_a_camera_without_onvif_releases_waiters(self) -> None:
        # nothing answers for it any more, such as after its client failed to build
        self.controller.cams = {}

        self._on_loop(self.controller.get_camera_status(CAMERA))

        self.assertTrue(self.controller.ptz_metrics[CAMERA].motor_stopped.is_set())


if __name__ == "__main__":
    unittest.main()
