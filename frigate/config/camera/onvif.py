from enum import Enum
from typing import Optional, Union

from pydantic import Field, field_validator, model_validator

from ..base import FrigateBaseModel
from ..env import EnvString
from .objects import DEFAULT_TRACKED_OBJECTS

__all__ = ["OnvifConfig", "PtzAutotrackConfig", "ZoomingModeEnum"]


class ZoomingModeEnum(str, Enum):
    disabled = "disabled"
    absolute = "absolute"
    relative = "relative"
    continuous = "continuous"


class PtzAutotrackConfig(FrigateBaseModel):
    enabled: bool = Field(default=False, title="Enable PTZ object autotracking.")
    calibrate_on_startup: bool = Field(
        default=False, title="Perform a camera calibration when Frigate starts."
    )
    zooming: ZoomingModeEnum = Field(
        default=ZoomingModeEnum.disabled,
        title="Autotracker zooming mode.",
        description="disabled: no zooming, absolute: position-based zoom, relative: concurrent zoom with pan/tilt, continuous: velocity-based zoom for cameras without position control"
    )
    zoom_factor: float = Field(
        default=0.3,
        title="Zooming factor (0.1-0.75).",
        ge=0.1,
        le=0.75,
    )
    track: list[str] = Field(default=DEFAULT_TRACKED_OBJECTS, title="Objects to track.")
    required_zones: list[str] = Field(
        default_factory=list,
        title="List of required zones to be entered in order to begin autotracking.",
    )
    return_preset: str = Field(
        default="home",
        title="Name of camera preset to return to when object tracking is over.",
    )
    timeout: int = Field(
        default=10, title="Seconds to delay before returning to preset."
    )
    continuous_speed: float = Field(
        default=2.0,
        title="Movement speed for ContinuousMove PTZ cameras (FOV units/sec).",
        description="How many FOV units the camera traverses per second at velocity=1.0. Higher values mean faster movement. Typical range: 0.5-4.0",
        ge=0.1,
        le=10.0,
    )
    continuous_zoom_speed: float = Field(
        default=1.0,
        title="Zoom speed for ContinuousMove PTZ cameras (zoom units/sec).",
        description="How fast the camera zooms at velocity=1.0. Lower values mean slower, more precise zoom. Typical range: 0.5-2.0",
        ge=0.1,
        le=5.0,
    )
    assumed_zoom_range: Optional[tuple[float, float]] = Field(
        default=None,
        title="Camera's zoom range in native units (e.g., optical zoom multiplier).",
        description="For cameras that don't report zoom position, specify the zoom range in the camera's native units as shown on OSD. Example: [1, 30] for a 30x optical zoom camera, or [1, 20] if camera can only reach 20x.",
    )
    preset_zoom_level: Optional[float] = Field(
        default=None,
        title="Zoom level at the return preset in native units.",
        description="For cameras that don't report zoom level, specify the zoom level at your preset position in the camera's native units (as shown on OSD). Example: 10 for a preset at 10x optical zoom, or 5.5 for 5.5x zoom.",
    )
    movement_weights: Optional[Union[str, list[str]]] = Field(
        default_factory=list,
        title="Internal value used for PTZ movements based on the speed of your camera's motor.",
    )
    enabled_in_config: Optional[bool] = Field(
        default=None, title="Keep track of original state of autotracking."
    )

    @field_validator("assumed_zoom_range", mode="before")
    @classmethod
    def validate_zoom_range(cls, v):
        if v is None:
            return None

        if isinstance(v, (list, tuple)) and len(v) == 2:
            min_val, max_val = v
            if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                if min_val > 0 and min_val < max_val:
                    return (float(min_val), float(max_val))
                else:
                    raise ValueError("assumed_zoom_range values must be positive with min < max (e.g., [1, 30] for 1x-30x zoom)")

        raise ValueError("assumed_zoom_range must be a tuple of two numbers representing min and max zoom in camera units")

    @model_validator(mode='after')
    def validate_preset_within_range(self):
        """Validate that preset_zoom_level is within assumed_zoom_range if both are set."""
        if self.preset_zoom_level is not None and self.assumed_zoom_range is not None:
            min_zoom, max_zoom = self.assumed_zoom_range
            if not (min_zoom <= self.preset_zoom_level <= max_zoom):
                raise ValueError(
                    f"preset_zoom_level ({self.preset_zoom_level}) must be within "
                    f"assumed_zoom_range ({min_zoom}, {max_zoom})"
                )
        return self

    @field_validator("movement_weights", mode="before")
    @classmethod
    def validate_weights(cls, v):
        if v is None:
            return None

        if isinstance(v, str):
            weights = list(map(str, map(float, v.split(","))))
        elif isinstance(v, list):
            weights = [str(float(val)) for val in v]
        else:
            raise ValueError("Invalid type for movement_weights")

        if len(weights) != 6:
            raise ValueError(
                "movement_weights must have exactly 6 floats, remove this line from your config and run autotracking calibration"
            )

        return weights


class OnvifConfig(FrigateBaseModel):
    host: EnvString = Field(default="", title="Onvif Host")
    port: int = Field(default=8000, title="Onvif Port")
    user: Optional[EnvString] = Field(default=None, title="Onvif Username")
    password: Optional[EnvString] = Field(default=None, title="Onvif Password")
    tls_insecure: bool = Field(default=False, title="Onvif Disable TLS verification")
    autotracking: PtzAutotrackConfig = Field(
        default_factory=PtzAutotrackConfig,
        title="PTZ auto tracking config.",
    )
    ignore_time_mismatch: bool = Field(
        default=False,
        title="Onvif Ignore Time Synchronization Mismatch Between Camera and Server",
    )
    sunba_quirks: bool = Field(
        default=False,
        title="Enable Sunba camera compatibility workarounds for broken PTZ implementation",
    )
