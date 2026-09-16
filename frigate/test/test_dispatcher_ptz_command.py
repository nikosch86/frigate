"""Tests for Dispatcher PTZ payload parsing, including set_return_preset."""

import unittest
from unittest.mock import MagicMock, patch

from frigate.comms.dispatcher import Dispatcher
from frigate.ptz.onvif import OnvifCommandEnum


def _build_dispatcher() -> Dispatcher:
    config = MagicMock()
    config.cameras = {"front": MagicMock()}
    with (
        patch("frigate.comms.dispatcher.CameraActivityManager"),
        patch("frigate.comms.dispatcher.AudioActivityManager"),
    ):
        return Dispatcher(config, MagicMock(), MagicMock(), {}, [])


class TestOnPtzCommand(unittest.TestCase):
    def setUp(self) -> None:
        self.dispatcher = _build_dispatcher()
        self.handle_command = self.dispatcher.onvif.handle_command

    def test_set_return_preset_routes_to_dedicated_command(self) -> None:
        self.dispatcher._on_ptz_command("front", "set_return_preset")

        self.handle_command.assert_called_once_with(
            "front", OnvifCommandEnum.set_return_preset, "return_preset"
        )

    def test_set_return_preset_is_case_insensitive(self) -> None:
        self.dispatcher._on_ptz_command("front", "SET_RETURN_PRESET")

        self.handle_command.assert_called_once_with(
            "front", OnvifCommandEnum.set_return_preset, "return_preset"
        )

    def test_set_return_preset_accepts_bytes_payload(self) -> None:
        self.dispatcher._on_ptz_command("front", b"set_return_preset")

        self.handle_command.assert_called_once_with(
            "front", OnvifCommandEnum.set_return_preset, "return_preset"
        )

    def test_plain_preset_still_routes_to_preset(self) -> None:
        self.dispatcher._on_ptz_command("front", "preset_Home")

        self.handle_command.assert_called_once_with(
            "front", OnvifCommandEnum.preset, "home"
        )

    def test_move_relative_keeps_its_params(self) -> None:
        self.dispatcher._on_ptz_command("front", "move_relative_0.1_-0.2")

        self.handle_command.assert_called_once_with(
            "front", OnvifCommandEnum.move_relative, "relative_0.1_-0.2"
        )

    def test_simple_command_has_empty_param(self) -> None:
        self.dispatcher._on_ptz_command("front", "move_left")

        self.handle_command.assert_called_once_with(
            "front", OnvifCommandEnum.move_left, ""
        )

    def test_unknown_command_is_ignored(self) -> None:
        self.dispatcher._on_ptz_command("front", "spin_around")

        self.handle_command.assert_not_called()
