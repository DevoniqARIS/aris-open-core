from __future__ import annotations

import argparse
import asyncio
from app.adapters.px4_mavsdk import PX4MavsdkDrone, PX4MavsdkConfig


async def main(args):
    drone = PX4MavsdkDrone(config=PX4MavsdkConfig(system_address=args.system_address))
    print(f"Connecting to PX4 via {args.system_address} ...")
    await drone.connect()
    t = await drone.telemetry()
    print("PX4 connection: PASS")
    print(f"GPS: {'OK' if t.gps_ok else 'NOT OK'}")
    print(f"Battery: {t.battery_percent:.1f}%")
    print(f"Position: {t.lat:.7f}, {t.lon:.7f}")
    print(f"Relative altitude: {t.altitude_m:.1f} m")
    print(f"Airborne: {t.airborne}")

    if not args.flight_test:
        print("Read-only smoke test complete. No arm/takeoff command was sent.")
        return

    if not args.i_accept_flight_risk:
        raise SystemExit(
            "--flight-test also requires --i-accept-flight-risk. Use PX4 SITL or a controlled test site."
        )
    if t.airborne:
        raise SystemExit("Refusing flight test because vehicle is already airborne")
    print(f"Flight test: takeoff to {args.takeoff_altitude:.1f} m")
    try:
        await drone.takeoff(args.takeoff_altitude)
        print("Takeoff: PASS")
        await asyncio.sleep(args.hold_seconds)
    finally:
        print("Landing...")
        await drone.land()
    print("Landing confirmed: PASS")


def parse_args():
    p = argparse.ArgumentParser(description="ARIS PX4/MAVSDK connectivity and SITL smoke test")
    p.add_argument("--system-address", default="udpin://0.0.0.0:14540")
    p.add_argument("--flight-test", action="store_true")
    p.add_argument("--i-accept-flight-risk", action="store_true")
    p.add_argument("--takeoff-altitude", type=float, default=5.0)
    p.add_argument("--hold-seconds", type=float, default=3.0)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
