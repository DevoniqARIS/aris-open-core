# PX4 integration status

ARIS Open Core v0.2.0 includes a real MAVSDK adapter for PX4/MAVLink systems.

## Internally validated
- adapter construction and capability registration
- MAVSDK connection/health flow using a deterministic System-compatible fake
- normalized GPS/battery/airborne telemetry
- takeoff sequence
- relative-altitude to AMSL conversion for `goto_location`
- arrival confirmation logic
- hold / RTL / land / abort paths
- camera shutter plus evidence-file download path
- MissionPlan / Capture schema support for `px4`
- live-adapter recovery-lock behavior
- full open-core regression suite

## Not claimed yet
- physical PX4 airframe validation
- PX4 SITL end-to-end transport execution in this build environment
- obstacle-sensor/perception integration from a real aircraft
- production camera compatibility across arbitrary MAVLink camera payloads
- certification or safety-critical deployment readiness

The repository includes a safe read-only connectivity test and an explicit opt-in
SITL/controlled-flight smoke test for users with PX4 available.
