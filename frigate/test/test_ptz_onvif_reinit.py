"""Tests for re-initializing a camera's ONVIF client after a config save.

The autotracker can be polling the camera for its motor to stop while the UI saves
its ONVIF settings. A status call on a closed session fails without yielding, so a
poll against the old session spun forever and froze the whole ONVIF loop.
"""

import asyncio
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
from onvif import ONVIFError

from frigate.camera import PTZMetrics
from frigate.ptz.onvif import OnvifController

CAMERA = "ptz"
MOVING = SimpleNamespace(
    MoveStatus=SimpleNamespace(PanTilt="MOVING", Zoom="IDLE"), Position=None
)


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
        self.calls_on_closed_session = 0
        self.aborted_requests = 0

        async def get_status(request):
            # a closed aiohttp session raises before awaiting anything
            if self.old_session_closed:
                self.calls_on_closed_session += 1
                raise RuntimeError("Session is closed")

            await asyncio.sleep(0.02)

            # closing the session fails the request that is still in flight
            if self.old_session_closed:
                self.aborted_requests += 1
                raise aiohttp.ServerDisconnectedError()

            return MOVING

        async def close_old_session():
            # the real ONVIFCamera.close() yields while it closes each session
            self.old_session_closed = True
            await asyncio.sleep(0.05)

        self.old_client = MagicMock(close=AsyncMock(side_effect=close_old_session))
        controller.cams = {
            CAMERA: {
                "onvif": self.old_client,
                "init": True,
                "active": True,
                "features": [],
                "ptz": SimpleNamespace(GetStatus=get_status),
                "status_request": MagicMock(),
            }
        }

        self.new_client = MagicMock(close=AsyncMock())
        self.init_calls: list = []

        async def install_new_client(cam_name):
            self.controller.cams[cam_name] = {"onvif": self.new_client, "init": False}
            return True

        async def new_client_cannot_connect_yet(camera_name):
            self.init_calls.append(self.controller.cams[camera_name]["onvif"])
            await asyncio.sleep(0.01)
            return False

        controller._init_single_camera = install_new_client
        controller._init_onvif = new_client_cannot_connect_yet

    def _stop_loop(self) -> None:
        # release a poll left spinning by a failure, so it cannot slow other tests
        self.controller.stopping = True
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

    def _start_waiting(self):
        wait = asyncio.run_coroutine_threadsafe(
            self._wait_for_motor(), self.controller.loop
        )
        time.sleep(0.05)
        return wait

    def _reinit(self) -> None:
        # a frozen loop never finishes the re-init
        self._on_loop(self.controller._reinit_camera(CAMERA))

    def test_reinit_finishes_and_the_poll_reaches_the_new_client(self) -> None:
        self._start_waiting()

        self._reinit()

        self.old_client.close.assert_awaited_once()
        self.assertIs(self.controller.cams[CAMERA]["onvif"], self.new_client)
        self.assertIn(self.new_client, self.init_calls)
        self.new_client.close.assert_not_awaited()
        self.assertEqual(self.calls_on_closed_session, 0)

    def test_no_status_request_is_aborted_by_the_close(self) -> None:
        # an aborted request is reported as "does not support the ONVIF GetStatus
        # method", and a body read cut off by the close can hang for a minute
        self._start_waiting()

        self._reinit()

        self.assertEqual(self.aborted_requests, 0)

    def test_save_while_idle_connects_the_new_client(self) -> None:
        # nothing else connects it, so the autotracker's next moves would fail
        self.controller.ptz_metrics[CAMERA].motor_stopped.set()

        self._reinit()

        self.assertEqual(self.init_calls, [self.new_client])

    def test_new_client_that_cannot_connect_releases_the_poll(self) -> None:
        # bad settings, such as a wrong port: retrying at once flooded the log
        async def refused(camera_name):
            self.init_calls.append(self.controller.cams[camera_name]["onvif"])
            return False

        self.controller._init_onvif = refused
        wait = self._start_waiting()

        self._reinit()

        wait.result(timeout=2)
        self.assertLessEqual(len(self.init_calls), 3)

    def test_init_error_releases_the_poll(self) -> None:
        async def unsupported(camera_name):
            raise ONVIFError("Device doesn't support service: media")

        self.controller._init_onvif = unsupported
        wait = self._start_waiting()

        self._reinit()

        # the error is reported, not raised into the autotracker's task
        self.assertIsNone(wait.result(timeout=2))

    def test_removing_onvif_releases_the_poll(self) -> None:
        self.controller.config.cameras[CAMERA].onvif.host = None
        wait = self._start_waiting()

        self._reinit()

        self.old_client.close.assert_awaited_once()
        self.assertNotIn(CAMERA, self.controller.cams)
        wait.result(timeout=2)

    def test_a_second_save_before_the_new_client_connects(self) -> None:
        clients = [MagicMock(close=AsyncMock()), MagicMock(close=AsyncMock())]

        async def install_next_client(cam_name):
            self.controller.cams[cam_name] = {"onvif": clients[0], "init": False}
            clients.reverse()
            return True

        self.controller._init_single_camera = install_next_client
        self._start_waiting()
        self._reinit()
        first_new = self.controller.cams[CAMERA]["onvif"]

        self._reinit()

        first_new.close.assert_awaited_once()
        self.assertIsNot(self.controller.cams[CAMERA]["onvif"], first_new)

    def test_status_of_a_camera_without_onvif_releases_waiters(self) -> None:
        # nothing answers for it any more, such as after its client failed to build
        self.controller.cams = {}

        self._on_loop(self.controller.get_camera_status(CAMERA))

        self.assertTrue(self.controller.ptz_metrics[CAMERA].motor_stopped.is_set())


class TestShutdownDuringReinit(unittest.TestCase):
    def test_old_session_is_closed_when_shutdown_cancels_the_reinit(self) -> None:
        # a shutdown right after a settings save cancels the poll mid re-init
        controller = OnvifController.__new__(OnvifController)
        controller.loop = asyncio.new_event_loop()
        controller.loop_thread = threading.Thread(
            target=controller.loop.run_forever, daemon=True
        )
        controller.loop_thread.start()
        controller.config_subscriber = MagicMock()
        controller.config = MagicMock()
        controller.config.cameras = {CAMERA: MagicMock()}
        controller.camera_configs = {CAMERA: controller.config.cameras[CAMERA]}
        controller.failed_cams = {}
        controller.status_locks = {}
        closing = threading.Event()

        async def slow_first_close():
            # the first close is the re-init's, which the shutdown interrupts
            if not closing.is_set():
                closing.set()
                await asyncio.sleep(3600)

        old_client = MagicMock(close=AsyncMock(side_effect=slow_first_close))
        controller.cams = {CAMERA: {"onvif": old_client, "init": True}}

        async def install_new_client(cam_name):
            controller.cams[cam_name] = {"onvif": MagicMock(close=AsyncMock())}
            return True

        controller._init_single_camera = install_new_client
        controller._init_onvif = AsyncMock(return_value=True)

        async def start_reinit() -> asyncio.Task:
            return asyncio.create_task(controller._reinit_camera(CAMERA))

        controller.config_poll = asyncio.run_coroutine_threadsafe(
            start_reinit(), controller.loop
        ).result(timeout=5)
        self.assertTrue(closing.wait(timeout=5))

        controller.close()

        # the interrupted close does not count, so the old session needs a second one
        self.assertEqual(old_client.close.await_count, 2)


if __name__ == "__main__":
    unittest.main()
