from app.adapters.factory import create_adapter
from app.adapters.px4_mavsdk import PX4MavsdkDrone


def test_px4_factory():
    d=create_adapter("px4", system_factory=lambda: object())
    assert isinstance(d, PX4MavsdkDrone)
