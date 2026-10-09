# PX4 / MAVSDK adapter

ARIS Open Core includes a direct PX4 adapter using the official MAVSDK Python gRPC client.
No mobile phone or intermediary app is required:

```text
ARIS -> PX4MavsdkDrone -> MAVSDK -> MAVLink -> PX4 autopilot
```

## Install

```bash
python -m pip install -r requirements-px4.txt
```

## Safe connectivity test

Start PX4 SITL (or connect a PX4 vehicle through your normal MAVLink transport), then run:

```bash
python -m tools.px4_smoke_test --system-address udpin://0.0.0.0:14540
```

By default the smoke test **does not arm or fly**. It only verifies connection, health,
home position, GPS position, battery, and airborne state.

## Opt-in SITL flight test

Use this only in PX4 SITL or a properly controlled test environment:

```bash
python -m tools.px4_smoke_test \
  --system-address udpin://0.0.0.0:14540 \
  --flight-test --takeoff-altitude 5
```

The flight test arms, climbs, holds briefly, and lands. It does not perform autonomous
inspection or obstacle avoidance. Those layers require additional integration with the
vehicle's actual perception/obstacle data and must be field validated.

## Adapter behavior

The adapter currently implements:

- MAVSDK connection and health gating
- normalized ARIS telemetry
- arm + takeoff to relative altitude
- global `goto_location` with ARIS relative-altitude to AMSL conversion
- current-speed request
- hold
- return-to-launch
- land with landing confirmation
- abort via RTL with land fallback
- camera shutter + evidence download when MAVSDK reports a `file_url`

ARIS uses relative-to-home altitude internally. MAVSDK `goto_location` uses absolute
altitude (AMSL), so the adapter obtains the home AMSL altitude and performs the conversion.

## Important safety boundary

The existence of the adapter does **not** mean ARIS is field-certified. Real-aircraft
command timing, failsafes, coordinate-frame assumptions, RTL behavior, camera support,
MAVLink transport, obstacle sensing, and airspace/legal requirements must be validated
on the actual aircraft before autonomous operation.
