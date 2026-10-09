# Devoniq ARIS Open Core v0.2

**ARIS is an open mission-intelligence layer for drone autonomy research.**

When given a high-level objective, ARIS translates it into a structured Mission IR, validates
it against deterministic safety rules, and executes it in a generic 3D simulation.

Example objective:

> Take off to 12 m. Find the nearest building. Fly to it while avoiding obstacles.
> Circle the building, return home, and land.

The public core version is designed to make the architecture inspectable and hackable.

**v0.2 adds a direct PX4/MAVSDK adapter**, so developers can connect ARIS to PX4/MAVLink systems without a phone app or proprietary bridge.

The public core is designed to make the architecture inspectable and hackable:
natural-language mission interpretation, Mission IR, safety validation, clean-slate
object discovery, 2D/3D planning, replanning, simulation adapters, and the browser
Simulation Lab.

## What this is

- A research/developer project.
- A generic rigid-body Bullet simulation.
- A hardware-agnostic adapter architecture.
- An **Apache-2.0 licensed** community core.

## What this is not

- Not a claim of real-aircraft validation. That is being planned.
- Not manufacturer firmware.
- Not certified obstacle avoidance.
- Not a safety-critical flight controller.
- Production hardware adapters are intentionally excluded from this repository.

## 5-minute start

### Windows

Install **Python 3.11 x64**, clone the repo, then double-click:

`START_ARIS_WINDOWS.bat`

The script creates a local virtual environment, installs the simulator dependencies,
runs a self-test, and starts the Simulation Lab.

### macOS / Linux

```bash
./start_aris.sh
```

Then open the local URL printed in the terminal.

## PX4 / MAVSDK

Install the optional PX4 dependency:

```bash
python -m pip install -r requirements-px4.txt
```

Run a read-only connectivity test:

```bash
python -m tools.px4_smoke_test --system-address udpin://0.0.0.0:14540
```

See [docs/PX4_MAVSDK.md](docs/PX4_MAVSDK.md) for SITL and real-aircraft integration notes.

## Live natural language

The simulator can run without an LLM connection, but live free-form mission
interpretation requires an OpenAI API key and a configured model:

```bash
cp .env.example .env.local
```

Fill in:

```text
OPENAI_API_KEY=...
OPENAI_MISSION_MODEL=...
```

Do **not** commit `.env.local`.

The commercial version of ARIS will run on proprietary LLMs based on use case and scenarios. 

## Verify the core without launching Bullet

```bash
python -m pip install -r requirements-dev.txt
python -m tools.community_self_test
python -m pytest
```

The public test suite exercises Mission IR validation, safety rules, general
natural-language schema handling, clean-slate discovery semantics, 3D planning,
progress-aware navigation contracts, asset coverage logic, and regression checks.

## Architecture

```text
Human objective
      |
      v
Mission interpreter
      |
      v
ARIS Mission IR
      |
      v
Deterministic Safety Gate
      |
      v
World discovery / target selection
      |
      v
2D / 3D planner + replanning
      |
      v
Drone adapter interface
      |
      v
Bullet Simulation Lab
```

The language model interprets intent. It does **not** directly issue arbitrary
motor commands. ARIS owns safety validation, bounded mission semantics, route
generation, and execution.

## Open-core boundary

See [OPEN_CORE_BOUNDARY.md](OPEN_CORE_BOUNDARY.md).

In short, this repository contains the developer/research core. Devoniq's commercial
layer is intended to cover production aircraft integrations, fleet/cloud operations,
production perception, enterprise integrations, deployment tooling, and support.

## Safety scope

ARIS Open Core is intended for civilian inspection, research, documentation, and
general autonomy experimentation. It does not include weaponization, targeting,
or autonomous pursuit of people. Dual-use and military infrastructure is reserved for 
Devoniq ARIS commercial versions.

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Project status

Early-stage research software. Expect rapid iteration. Issues, reproducible bug
reports, tests, and pull requests are welcome.
