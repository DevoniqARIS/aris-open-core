# ARIS Open-Core Boundary

The goal is simple: make the public project genuinely useful while preserving a
clear path for Devoniq to build a commercial product.

## Apache-2.0 public core

The public repository includes:

- ARIS Mission IR and validation models
- live natural-language mission interpretation interface
- deterministic safety gate
- generic drone adapter interface and capability model
- open PX4/MAVSDK adapter for community/SITL integration
- in-memory/replay adapters
- clean-slate world-discovery semantics
- 2D and 3D route planning
- obstacle-aware replanning
- progress/recovery logic
- generic Bullet Simulation Lab
- browser mission-control UI
- evidence/quality primitives used by the demo
- unit and regression tests
- examples and developer documentation

This is enough for a developer to clone the project, run the simulator, inspect the
architecture, experiment with missions, and build their own adapter.

## Devoniq commercial layer

The following should remain outside the public repository unless Devoniq later
chooses otherwise:

- production-ready DJI/Parrot/ArduPilot adapters and field-hardened PX4 deployment tooling
- manufacturer-specific certification/validation work
- hardened real-aircraft obstacle avoidance
- production computer-vision models and deployment pipelines
- persistent fleet-wide asset intelligence at scale
- enterprise cloud control plane and fleet orchestration
- customer data connectors and enterprise integrations
- RBAC, SSO, audit/compliance administration
- managed mission history, analytics, alerting, and SLA infrastructure
- field-calibration tooling and hardware qualification
- commercial support, deployment, and integration packages

The distinction is not "demo code versus useful code." The public core should be
useful. The commercial product earns money by making the same architecture
production-grade, connected to real fleets, deployable, supportable, and scalable.
