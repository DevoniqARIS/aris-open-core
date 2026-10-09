"""Run one local simulation server and open its authenticated browser UI."""
import argparse, os, secrets, threading, webbrowser, faulthandler, sys, time
from pathlib import Path
import uvicorn
from .server import create_app

def _load_local_env():
    path = Path(__file__).resolve().parents[1] / '.env.local'
    if not path.is_file():
        return
    for raw in path.read_text(encoding='utf-8-sig').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)

def main():
    _load_local_env()
    # Preserve a Python/all-thread dump for native crashes (including Bullet faults).
    # This is intentionally outside per-run folders so it survives abrupt process exit.
    runtime = Path(__file__).resolve().parents[1] / "demo_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    fatal_log = (runtime / "backend_fatal.log").open("a", encoding="utf-8")
    fatal_log.write(f"\n--- ARIS backend start {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    fatal_log.flush()
    faulthandler.enable(fatal_log, all_threads=True)
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--listen',default='127.0.0.1')
    parser.add_argument('--no-browser',action='store_true')
    args=parser.parse_args()
    token=secrets.token_urlsafe(32)
    local=f'http://127.0.0.1:{args.port}/#token={token}'
    print('\nARIS Simulation Lab — generic rigid-body physics; not Parrot firmware.\n')
    print('Open this private session link on this computer:\n'+local+'\n',flush=True)
    if args.listen!='127.0.0.1':
        print('LAN mode: replace 127.0.0.1 with this computer’s LAN IP. Keep the token private. Do not expose this HTTP server to the public Internet.',flush=True)
    if not args.no_browser:threading.Timer(2,lambda:webbrowser.open(local)).start()
    uvicorn.run(create_app(token=token),host=args.listen,port=args.port,access_log=False)

if __name__=='__main__':main()
