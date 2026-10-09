"""Fast startup check for the native Windows ARIS Simulation Lab."""
from __future__ import annotations

import platform
import tempfile
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    print(f"       OS: {platform.platform()}")
    try:
        import pybullet as p
        print(f"       PyBullet API: {p.getAPIVersion()}")
    except Exception as exc:
        print(f"ERROR importing pybullet: {type(exc).__name__}: {exc}")
        return 2

    try:
        from simulation.world import World, HOME
        with tempfile.TemporaryDirectory(prefix="aris_native_check_") as td:
            world = World(Path(td) / "images", scenario="adaptive_complex", rate=8)
            world.start()
            state = world.state()
            rgb, _ = world.render(width=160, height=120)
            sensed = world.scan_obstacles(HOME[:2], range_m=10.0, altitude_m=9.0)
            corner_markers = [m for m in world.markers if str(m.get("label", "")).startswith("corner_")]
            ok = (
                tuple(rgb.shape) == (120, 160, 3)
                and len(state.get("position", [])) == 3
                and len(world.nav_obstacles) >= 7
                and len(sensed) >= 1
                and len(corner_markers) == 8
            )
            world.close()
        if not ok:
            print("ERROR: adaptive physics/camera/obstacle smoke check returned an unexpected result")
            return 3
    except Exception as exc:
        print(f"ERROR running physics/camera smoke check: {type(exc).__name__}: {exc}")
        return 4

    print("       Physics + rendered camera + adaptive obstacle scene: OK")
    print("       Native Windows simulation check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
