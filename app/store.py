import sqlite3, threading
from pathlib import Path
from .models import Asset, MissionPlan, Observation, SecurityEvent, MissionStatus
from .completion_engine import CompletionState


class Store:
    def __init__(self, path=Path("runtime/aris_v2.db")):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS assets(id TEXT PRIMARY KEY,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS missions(id TEXT PRIMARY KEY,asset_id TEXT NOT NULL,status TEXT NOT NULL,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS observations(id TEXT PRIMARY KEY,mission_id TEXT NOT NULL,asset_id TEXT NOT NULL,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS security_events(id TEXT PRIMARY KEY,mission_id TEXT NOT NULL,asset_id TEXT NOT NULL,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS completion_states(mission_id TEXT PRIMARY KEY,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS control_lock(name TEXT PRIMARY KEY,mission_id TEXT NOT NULL);
        """)
        self.conn.commit()

    def _write(self, query, args):
        with self.lock, self.conn:
            self.conn.execute(query, args)

    def _one(self, query, args):
        with self.lock:
            return self.conn.execute(query, args).fetchone()

    def _all(self, query, args):
        with self.lock:
            return self.conn.execute(query, args).fetchall()

    def save_asset(self, x):
        self._write(
            "INSERT OR REPLACE INTO assets VALUES (?,?)", (x.id, x.model_dump_json())
        )

    def get_asset(self, id):
        r = self._one("SELECT payload FROM assets WHERE id=?", (id,))
        return Asset.model_validate_json(r["payload"]) if r else None

    def save_mission(self, x):
        self._write(
            "INSERT OR REPLACE INTO missions VALUES (?,?,?,?)",
            (x.id, x.asset_id, x.status.value, x.model_dump_json()),
        )

    def get_mission(self, id):
        r = self._one("SELECT payload FROM missions WHERE id=?", (id,))
        return MissionPlan.model_validate_json(r["payload"]) if r else None

    def save_observation(self, x):
        self._write(
            "INSERT INTO observations VALUES (?,?,?,?)",
            (x.id, x.mission_id, x.asset_id, x.model_dump_json()),
        )

    def observations_for_mission(self, id):
        return [
            Observation.model_validate_json(r["payload"])
            for r in self._all(
                "SELECT payload FROM observations WHERE mission_id=? ORDER BY rowid",
                (id,),
            )
        ]

    def missions_for_asset(self, id):
        return [
            MissionPlan.model_validate_json(r["payload"])
            for r in self._all(
                "SELECT payload FROM missions WHERE asset_id=? ORDER BY rowid", (id,)
            )
        ]

    def save_security_event(self, x):
        self._write(
            "INSERT INTO security_events VALUES (?,?,?,?)",
            (x.id, x.mission_id, x.asset_id, x.model_dump_json()),
        )

    def security_events_for_mission(self, id):
        return [
            SecurityEvent.model_validate_json(r["payload"])
            for r in self._all(
                "SELECT payload FROM security_events WHERE mission_id=?", (id,)
            )
        ]

    def save_completion_state(self, id, x):
        self._write(
            "INSERT OR REPLACE INTO completion_states VALUES (?,?)",
            (id, x.model_dump_json()),
        )

    def completion_state(self, id):
        r = self._one("SELECT payload FROM completion_states WHERE mission_id=?", (id,))
        return CompletionState.model_validate_json(r["payload"]) if r else None

    def claim(self, id):
        # SQLite lock covers multiple worker processes too. One aircraft execution at a time.
        with self.lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                p = self.get_mission(id)
                if not p or p.status != MissionStatus.PLANNED:
                    raise ValueError("Mission is not startable")
                if self.conn.execute("SELECT 1 FROM control_lock").fetchone():
                    raise ValueError("Aircraft controller busy or landing unconfirmed")
                self.conn.execute(
                    "INSERT INTO control_lock VALUES (?,?)", ("aircraft", id)
                )
                p.status = MissionStatus.QUEUED
                self.conn.execute(
                    "UPDATE missions SET status=?,payload=? WHERE id=?",
                    (p.status.value, p.model_dump_json(), id),
                )
                self.conn.commit()
                return p
            except Exception:
                self.conn.rollback()
                raise

    def release(self, id):
        self._write("DELETE FROM control_lock WHERE mission_id=?", (id,))

    def recover_interrupted(self):
        # Use a single server worker. On process restart, never silently resume a flight.
        with self.lock:
            rows = self._all(
                "SELECT payload FROM missions WHERE status IN (?,?,?,?,?)",
                tuple(
                    s.value
                    for s in [
                        MissionStatus.QUEUED,
                        MissionStatus.VALIDATED,
                        MissionStatus.RUNNING,
                        MissionStatus.REINSPECTING,
                        MissionStatus.RETURNING_HOME,
                    ]
                ),
            )
            for row in rows:
                p = MissionPlan.model_validate_json(row["payload"])
                p.status = MissionStatus.FAILED
                p.error = "Controller restarted during execution; mission not resumed."
                if p.adapter in {"dji", "parrot"}:
                    p.recovery_error = "Aircraft state unknown; hardware control lock retained pending local operator recovery."
                else:
                    self.release(p.id)
                self.save_mission(p)

    def close(self):
        with self.lock:
            self.conn.close()
