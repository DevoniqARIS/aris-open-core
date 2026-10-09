# ARIS Open Core v0.2.0

## Added
- Direct PX4/MAVSDK adapter (`PX4MavsdkDrone`)
- MAVLink transport via official MAVSDK Python gRPC client
- PX4 capability profile
- normalized telemetry, takeoff, global go-to, hold, RTL, land and abort
- relative-altitude to AMSL conversion for `goto_location`
- optional MAVSDK camera shutter/evidence download using `CaptureInfo.file_url`
- safe read-only PX4 connectivity test CLI
- opt-in SITL flight smoke test
- PX4 integration documentation and optional dependency file

## Validation status
The adapter is software-tested with a deterministic MAVSDK-compatible fake backend.
The package does not claim physical-aircraft validation. A real PX4 or PX4 SITL instance
is still required to validate the transport and autopilot behavior end to end.
