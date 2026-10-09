from pathlib import Path
import os, sys, asyncio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def load_env():
    path = ROOT / ".env.local"
    if not path.exists():
        raise SystemExit("ERROR: .env.local not found")
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip().strip('"').strip("'")

async def main():
    load_env()
    key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MISSION_MODEL", "")
    print("ARIS GPT diagnostics")
    print("API key present:", bool(key), f"(length={len(key)})" if key else "")
    print("Mission model:", model or "<missing>")
    from app.maneuver import verify_gpt_link
    try:
        result = await verify_gpt_link()
        print("VERIFIED:", result)
    except Exception as e:
        print("FAILED:", type(e).__name__ + ":", str(e))
        raise SystemExit(1)

if __name__ == "__main__":
    asyncio.run(main())
