from pathlib import Path

from .sim import SimDrone
from .replay import ReplayDrone
from .px4_mavsdk import PX4MavsdkDrone, PX4MavsdkConfig


def create_adapter(name: str, **kwargs):
    name = name.lower().strip()
    if name == "sim":
        return SimDrone(**kwargs)
    if name == "replay":
        return ReplayDrone(**kwargs)
    if name in {"px4", "mavsdk", "mavlink"}:
        return PX4MavsdkDrone(**kwargs)
    raise ValueError(f"Unsupported ARIS adapter: {name}")
