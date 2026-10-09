# ARIS Open Core v0.1.0

First public community release.

Included:
- Apache-2.0 license
- Mission IR and natural-language interpretation interface
- deterministic safety validation
- clean-slate object-discovery semantics
- 2D/3D obstacle-aware planning and replanning
- generic Bullet Simulation Lab
- browser mission-control UI
- generic adapter interfaces
- offline self-test
- curated regression suite
- GitHub Actions CI

Intentionally excluded:
- production DJI/Parrot/PX4/ArduPilot adapters
- field-validated obstacle avoidance
- enterprise fleet/cloud control plane
- customer integrations and production deployment tooling

Validation for this package in the build environment:
- community self-test: PASS
- curated regression suite: 80 passed
- Python source compilation: PASS

Native PyBullet runtime was not executed in the packaging environment because PyBullet is not installed there. The repository includes the simulator dependency and startup scripts for a normal developer machine.
