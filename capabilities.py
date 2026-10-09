from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class DroneCapabilities(BaseModel):
    """Normalized capabilities exposed to ARIS regardless of manufacturer SDK."""

    model_config = ConfigDict(extra="forbid")

    adapter: str
    vendor: str
    model: str
    connection: str
    telemetry: bool = True
    takeoff_land: bool = True
    goto_position: bool = True
    local_xy_motion: bool = False
    heading_control: bool = False
    gimbal_control: bool = False
    still_capture: bool = False
    return_home: bool = False
    obstacle_sensing: bool = False
    obstacle_map: bool = False
    object_detection: bool = False
    max_speed_mps: float | None = Field(default=None, gt=0)
    max_altitude_m: float | None = Field(default=None, gt=0)
    validation: Literal["simulation", "software_only", "physical"] = "software_only"
    notes: list[str] = []


PROFILES = {
    "px4": DroneCapabilities(
        adapter="px4", vendor="PX4/MAVLink", model="MAVSDK-compatible autopilot",
        connection="MAVSDK over MAVLink", heading_control=True,
        still_capture=True, return_home=True,
        max_speed_mps=4, max_altitude_m=30, validation="software_only",
        notes=[
            "Open-core MAVSDK adapter included.",
            "Flight/telemetry paths are software-tested; real-aircraft validation remains the operator's responsibility.",
            "Camera evidence download requires a MAVSDK camera that reports a retrievable file URL."
        ],
    ),
    "physics": DroneCapabilities(
        adapter="physics", vendor="Generic", model="Bullet rigid-body simulator",
        connection="in-process", local_xy_motion=True, heading_control=True,
        gimbal_control=True, still_capture=True, return_home=True,
        obstacle_sensing=True, obstacle_map=True, object_detection=True, max_speed_mps=5,
        max_altitude_m=20, validation="simulation",
        notes=["Idealized position/yaw control; not manufacturer firmware."],
    ),
    "sim": DroneCapabilities(
        adapter="sim", vendor="Generic", model="In-memory test stub",
        connection="in-process", still_capture=True, return_home=True,
        max_speed_mps=4, max_altitude_m=30, validation="simulation",
        notes=["Kinematic software stub; not physics or hardware."],
    ),
    "parrot": DroneCapabilities(
        adapter="parrot", vendor="Parrot", model="external adapter target",
        connection="not included in open core", heading_control=True,
        gimbal_control=True, still_capture=True, return_home=True,
        max_speed_mps=4, max_altitude_m=30, validation="software_only",
        notes=["Production hardware adapter implementation is not included in ARIS Open Core."],
    ),
    "dji": DroneCapabilities(
        adapter="dji", vendor="DJI", model="external adapter target",
        connection="not included in open core", heading_control=True,
        gimbal_control=True, still_capture=True, return_home=True,
        max_speed_mps=4, max_altitude_m=30, validation="software_only",
        notes=["Production hardware adapter implementation is not included in ARIS Open Core."],
    ),
}


def capability_profile(adapter: str) -> DroneCapabilities:
    try:
        return PROFILES[adapter].model_copy(deep=True)
    except KeyError as exc:
        raise ValueError(f"No ARIS capability profile registered for adapter: {adapter}") from exc
