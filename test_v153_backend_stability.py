from pathlib import Path


def root():
    return Path(__file__).resolve().parents[1]


def test_object_visibility_is_guarded_by_world_lock_source_contract():
    text = (root() / "simulation" / "world.py").read_text(encoding="utf-8")
    start = text.index("    def object_visible")
    end = text.index("\n    def nav_box", start)
    block = text[start:end]
    assert "with self.lock:" in block
    assert "p.rayTest" in block


def test_arrival_has_progress_watchdog_source_contract():
    text = (root() / "simulation" / "adapter.py").read_text(encoding="utf-8")
    start = text.index("    async def _arrive")
    end = text.index("\n    async def takeoff", start)
    block = text[start:end]
    assert "last_progress" in block
    assert "Physics movement stopped converging" in block
    assert "Physics movement timed out" in block


def test_backend_traceback_artifact_is_preserved_source_contract():
    text = (root() / "simulation" / "server.py").read_text(encoding="utf-8")
    assert "backend_error.txt" in text
    assert "traceback.format_exc()" in text


def test_physics_worker_error_keeps_original_exception_message():
    text = (root() / "simulation" / "world.py").read_text(encoding="utf-8")
    assert "physics worker failed: {exc}" in text
