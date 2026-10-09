# Architecture

ARIS separates semantic interpretation from deterministic execution.

1. **Mission interpretation** converts a human objective into structured Mission IR.
2. **Normalization** fills only bounded, auditable defaults.
3. **Safety validation** rejects missions outside the configured envelope.
4. **Discovery** learns target objects and obstacles at runtime in adaptive mode.
5. **Planning** generates 2D/3D routes and alternate goal regions.
6. **Execution** sends bounded commands through a drone-adapter interface.
7. **Observation** updates the world model and can invalidate the active route.
8. **Replanning** chooses another route when new information makes the current route unsafe.
9. **Completion** records mission outcome and evidence/audit data.

The current public simulator uses generic Bullet physics and idealized control. It
is deliberately not presented as a manufacturer-specific flight model.


## PX4 path

`ARIS MissionExecutor -> DroneAdapter -> PX4MavsdkDrone -> MAVSDK -> MAVLink -> PX4`

The adapter converts ARIS relative altitude into MAVSDK AMSL altitude and normalizes PX4 telemetry back into ARIS models.
