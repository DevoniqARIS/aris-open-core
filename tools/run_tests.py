"""Reproducible test runner with machine-readable results and exact source digest."""

import contextlib, hashlib, importlib.metadata, io, json, platform, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pytest


class Recorder:
    def __init__(self):
        self.results = []

    def pytest_runtest_logreport(self, report):
        if report.when == "call" or report.failed:
            self.results.append(
                {
                    "test": report.nodeid,
                    "phase": report.when,
                    "outcome": report.outcome,
                    "duration_seconds": round(report.duration, 6),
                    "failure": str(report.longrepr) if report.failed else None,
                }
            )


def main():
    output = ROOT / "validation"
    output.mkdir(exist_ok=True)
    recorder = Recorder()
    buf = io.StringIO()
    started = datetime.now(timezone.utc).isoformat()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        code = pytest.main(
            [
                str(ROOT / "tests"),
                "-q",
                "--disable-warnings",
                "--junitxml=" + str(output / "pytest.xml"),
            ],
            plugins=[recorder],
        )
    versions = {
        p: importlib.metadata.version(p)
        for p in [
            "fastapi",
            "httpx",
            "pydantic",
            "Pillow",
            "numpy",
            "opencv-python-headless",
            "shapely",
            "pytest",
        ]
    }
    for optional in ["parrot-olympe", "PySDL2", "PyOpenGL"]:
        try:
            versions[optional] = importlib.metadata.version(optional)
        except importlib.metadata.PackageNotFoundError:
            versions[optional] = None
    digests = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for parent in ["app", "tests", "tools"]
        for p in sorted((ROOT / parent).rglob("*.py"))
    }
    result = {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        "exit_code": int(code),
        "passed": sum(r["outcome"] == "passed" for r in recorder.results),
        "failed": sum(r["outcome"] == "failed" for r in recorder.results),
        "skipped": sum(r["outcome"] == "skipped" for r in recorder.results),
        "results": recorder.results,
        "source_sha256": digests,
        "scope": "Local regression, failure injection, pixel operations on generated fixtures, mocked provider/gateway/Parrot transport. Parrot SDK contract test constructs real Olympe messages when installed (otherwise skips). No aircraft, Sphinx, live OpenAI, flight, or field-accuracy validation.",
    }
    (output / "results.json").write_text(json.dumps(result, indent=2))
    (output / "pytest-output.txt").write_text(buf.getvalue())
    print(buf.getvalue())
    print(
        json.dumps(
            {
                k: result[k]
                for k in ["passed", "failed", "skipped", "exit_code", "scope"]
            },
            indent=2,
        )
    )
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
