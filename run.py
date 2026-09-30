"""NeuraSight launcher.

Starts the three services that make up the application and tells you plainly
what is wrong when something will not start.

    python run.py                 start everything
    python run.py --check         run the preflight checks only, start nothing
    python run.py --no-frontend   API services only (no Vite)
    python run.py --install       install missing Node/Python deps first
    python run.py --kill          free ports 8000/5000/3000 and exit

Press Ctrl+C to stop everything.

Design notes
------------
* Child output streams straight to this console. An earlier version used
  ``stdout=PIPE`` without ever reading it, which hid every startup error and
  could deadlock once the pipe buffer filled.
* ``shell=True`` is avoided. With it, ``Popen.poll()`` watches the ``cmd.exe``
  wrapper rather than the server, so crash detection was unreliable in both
  directions. Executables are resolved explicitly instead.
* Readiness is decided by polling ``/health``, not by ``sleep(5)``. The brain
  ensemble loads ~645 MB of weights on first use, so a fixed sleep was always
  either too short or wasteful.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FASTAPI_DIR = ROOT / "backend" / "fastapi"
EXPRESS_DIR = ROOT / "backend" / "express"
FRONTEND_DIR = ROOT / "frontend"

FASTAPI_PORT = 8000
EXPRESS_PORT = 5000
FRONTEND_PORT = 3000

# Plain ASCII markers. Previously these were tick/cross/warning glyphs, which a
# cp1252 console cannot encode - that needed a runtime fallback, and it still
# rendered as mojibake whenever the encoding was forced to UTF-8 but the console
# codepage was not. Fixed-width labels also keep the report columns aligned.
OK, BAD, WARN, DOT = "[ ok ]", "[fail]", "[warn]", "  -  "


def say(msg: str = "") -> None:
    print(msg, flush=True)


def rule(title: str = "") -> None:
    say("=" * 64 if not title else f"\n{'=' * 64}\n  {title}\n{'=' * 64}")


# ---------------------------------------------------------------------------
# environment discovery
# ---------------------------------------------------------------------------

def find_python() -> str:
    """Prefer a project virtualenv, else the interpreter running this script."""
    candidates = [
        FASTAPI_DIR / "venv", FASTAPI_DIR / ".venv", ROOT / "venv", ROOT / ".venv",
    ]
    sub = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    for base in candidates:
        exe = base / sub
        if exe.exists():
            return str(exe)
    return sys.executable


def find_exe(name: str) -> str | None:
    """Resolve a Node tool to a real path (npm -> npm.cmd on Windows)."""
    found = shutil.which(name)
    if found:
        return found
    if os.name == "nt":
        for ext in (".cmd", ".exe", ".bat"):
            found = shutil.which(name + ext)
            if found:
                return found
    return None


def _listening(port: int) -> bool:
    """True if anything is listening on the port, over IPv4 *or* IPv6.

    Both families must be probed: Vite binds ``localhost``, which resolves to
    ``::1`` on this platform, so an IPv4-only probe reports the port free and
    the next start then dies with EADDRINUSE.
    """
    for family, host in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as probe:
                probe.settimeout(0.4)
                if probe.connect_ex((host, port)) == 0:
                    return True
        except OSError:
            continue  # family unavailable
    return False


def port_owner(port: int) -> int | None:
    """Return the PID listening on a port, or None. Best effort."""
    if not _listening(port):
        return None
    try:
        if os.name == "nt":
            # No `-p TCP`: that filter covers only IPv4, because Windows treats
            # IPv6 as a separate "TCPv6" protocol. Vite listens on ::1, so the
            # filtered form could not see it and the PID lookup silently failed,
            # leaving --kill unable to free port 3000. Plain `netstat -ano`
            # lists both families with "TCP" in the proto column.
            out = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True, text=True, timeout=15,
            ).stdout
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 5 and parts[0] == "TCP" \
                        and parts[1].rsplit(":", 1)[-1] == str(port) \
                        and parts[3] == "LISTENING":
                    return int(parts[4])
        else:
            out = subprocess.run(
                ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            if out:
                return int(out.splitlines()[0])
    except Exception:
        pass
    return -1  # in use, owner unknown


def kill_pid(pid: int) -> bool:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"],
                           capture_output=True, timeout=15)
        else:
            os.kill(pid, 9)
        return True
    except Exception:
        return False


def _http_once(url: str, timeout: float) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 500
    except urllib.error.HTTPError:
        return True  # responded, so the server is alive
    except Exception:
        return False


def http_ok(url: str, timeout: float = 2.0) -> bool:
    """Probe a health URL, tolerating an IPv4/IPv6 mismatch.

    uvicorn is bound to 127.0.0.1 while Vite binds ``localhost`` (``::1`` here),
    so a single fixed host makes one of the two look permanently unreachable.
    """
    if _http_once(url, timeout):
        return True
    swapped = (url.replace("127.0.0.1", "localhost") if "127.0.0.1" in url
               else url.replace("localhost", "127.0.0.1") if "localhost" in url
               else None)
    return bool(swapped) and _http_once(swapped, timeout)


def read_env_value(key: str) -> str | None:
    """Read a key from the repo-root .env without needing python-dotenv."""
    env = ROOT / ".env"
    if not env.exists():
        return None
    for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            if k.strip() == key:
                return v.strip()
    return None


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

def check_python(problems: list[str]) -> None:
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 10):
        problems.append(
            f"Python {major}.{minor} is too old. The pinned torch build needs 3.10-3.12."
        )
        say(f"  {BAD} Python {major}.{minor}")
    elif (major, minor) > (3, 12):
        say(f"  {WARN} Python {major}.{minor} - torch wheels may not exist for this version")
    else:
        say(f"  {OK} Python {major}.{minor}.{sys.version_info[2]}")


def check_py_packages(python: str, problems: list[str], auto_install: bool) -> None:
    needed = {
        "fastapi": "fastapi", "uvicorn": "uvicorn", "torch": "torch", "timm": "timm",
        "PIL": "pillow", "numpy": "numpy", "cv2": "opencv-python",
        "sklearn": "scikit-learn", "reportlab": "reportlab",
        "sentence_transformers": "sentence-transformers",
    }
    code = "".join(
        f"try:\n    import {mod}\nexcept Exception:\n    print('{mod}')\n"
        for mod in needed
    )
    try:
        out = subprocess.run([python, "-c", code], capture_output=True, text=True,
                             timeout=180).stdout.split()
    except Exception as exc:
        say(f"  {WARN} Could not probe Python packages: {exc}")
        return

    if not out:
        say(f"  {OK} Python packages ({len(needed)} checked)")
        return

    missing = sorted({needed[m] for m in out if m in needed})
    req = FASTAPI_DIR / "requirements.txt"
    if auto_install:
        say(f"  {WARN} Installing {len(missing)} missing package(s): {', '.join(missing)}")
        subprocess.run([python, "-m", "pip", "install", "-r", str(req)], cwd=FASTAPI_DIR)
        say(f"  {DOT} Re-run 'python run.py' to verify.")
    else:
        say(f"  {BAD} Missing Python packages: {', '.join(missing)}")
        problems.append(
            f"Install Python deps:  {python} -m pip install -r "
            f"{req.relative_to(ROOT)}\n      (or re-run with --install)"
        )


def check_node(problems: list[str], auto_install: bool, want_frontend: bool) -> None:
    npm = find_exe("npm")
    node = find_exe("node")
    if not node or not npm:
        problems.append("Node.js and npm are required. Install from https://nodejs.org (18+).")
        say(f"  {BAD} Node.js / npm not found on PATH")
        return
    try:
        ver = subprocess.run([node, "--version"], capture_output=True, text=True,
                             timeout=20).stdout.strip()
        say(f"  {OK} Node {ver}")
    except Exception:
        say(f"  {OK} Node found")

    # A node_modules directory is not proof of a working install: packages can be
    # present but gutted. Probe a real entry point instead.
    projects = [("Express", EXPRESS_DIR, "express/package.json")]
    if want_frontend:
        projects.append(("Frontend", FRONTEND_DIR, "vite/bin/vite.js"))

    for label, folder, sentinel in projects:
        target = folder / "node_modules" / Path(sentinel)
        if target.exists():
            say(f"  {OK} {label} node_modules")
            continue
        if auto_install:
            say(f"  {WARN} {label} deps missing - running npm install...")
            subprocess.run([npm, "install"], cwd=folder)
            if target.exists():
                say(f"  {OK} {label} node_modules restored")
            else:
                problems.append(
                    f"{label}: npm install did not restore {sentinel}. The install is "
                    f"likely corrupt. Delete {folder / 'node_modules'} and retry."
                )
        else:
            say(f"  {BAD} {label} node_modules missing or incomplete")
            problems.append(
                f"Install {label} deps:  cd {folder.relative_to(ROOT)} && npm install"
                f"\n      (or re-run with --install)"
            )


def check_env_files() -> None:
    pairs = [
        (ROOT / ".env", ROOT / ".env.example"),
        (EXPRESS_DIR / ".env", EXPRESS_DIR / ".env.example"),
        (FASTAPI_DIR / ".env", FASTAPI_DIR / ".env.example"),
    ]
    for env, example in pairs:
        if env.exists():
            continue
        if example.exists():
            shutil.copyfile(example, env)
            say(f"  {OK} created {env.relative_to(ROOT)} from .env.example")
        else:
            say(f"  {WARN} {env.relative_to(ROOT)} missing and no example to copy")


def check_weights() -> None:
    modules = {
        "brain_mri": (ROOT / "models", [
            "BRAIN_MRI_EFFICIENTNET.pth", "BRAIN_MRI_RESNET.pth",
            "BRAIN_MRI_DENSENET.pth", "BRAIN_MRI_VGG.pth",
        ], "meta_model.pkl"),
        "chest_xray": (ROOT / "chest", [
            "CHEST_XRAY_EFFICIENTNET.pth", "CHEST_XRAY_RESNET.pth",
            "CHEST_XRAY_DENSENET.pth",
        ], "meta_model_Chest_Xray.pkl"),
    }
    for name, (folder, weights, meta) in modules.items():
        missing = [w for w in weights if not (folder / w).exists()]
        if not missing and (folder / meta).exists():
            say(f"  {OK} {name}: {len(weights)} weights + meta-learner")
        elif len(missing) == len(weights):
            say(f"  {BAD} {name}: no weights in {folder.name}/ - this module will return 503")
        else:
            detail = ", ".join(missing) if missing else f"{meta} missing"
            say(f"  {WARN} {name}: degraded - {detail}")
            say(f"       Missing base models are replaced by zero vectors, which the "
                f"API reports via models_skipped.")


def check_artefacts() -> None:
    layers = [
        ("calibration + conformal", ROOT / "models/calibration/calibration.json",
         "probabilities will be reported uncalibrated with no prediction sets"),
        ("novelty detection", ROOT / "models/calibration/ood.json",
         "inputs will not be screened; validated will report false"),
        ("retrieval index", ROOT / "knowledge/index.npz",
         "evidence retrieval falls back to lexical matching"),
        ("clinical narratives", ROOT / "knowledge/narratives",
         "reports fall back to the deterministic template"),
    ]
    for label, path, consequence in layers:
        if path.exists():
            say(f"  {OK} {label}")
        else:
            say(f"  {WARN} {label} missing - {consequence}")


def check_mongo() -> None:
    uri = read_env_value("MONGODB_URI") or "mongodb://localhost:27017/neurasight"
    is_remote = uri.startswith("mongodb+srv://") or (
        "localhost" not in uri and "127.0.0.1" not in uri
    )
    if is_remote:
        say(f"  {DOT} MongoDB is remote - skipping reachability probe")
        return
    host, port = "127.0.0.1", 27017
    try:
        hostpart = uri.split("://", 1)[1].split("/")[0].split("@")[-1]
        if ":" in hostpart:
            host, port_s = hostpart.rsplit(":", 1)
            port = int(port_s)
            host = "127.0.0.1" if host == "localhost" else host
    except Exception:
        pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1.5)
        if probe.connect_ex((host, port)) == 0:
            say(f"  {OK} MongoDB reachable on {host}:{port}")
        else:
            say(f"  {WARN} MongoDB not reachable on {host}:{port}")
            say(f"       Scans still run, but history will not persist "
                f"(the API returns saved: false).")


def check_ports(ports: list[int], auto_kill: bool, problems: list[str]) -> None:
    for port in ports:
        pid = port_owner(port)
        if pid is None:
            say(f"  {OK} port {port} free")
            continue
        label = f"PID {pid}" if pid and pid > 0 else "unknown process"
        if auto_kill:
            if kill_pid(pid) if pid and pid > 0 else False:
                time.sleep(1.5)
                say(f"  {OK} port {port} freed (killed {label})")
            else:
                problems.append(f"Port {port} is in use by {label} and could not be freed.")
                say(f"  {BAD} port {port} in use by {label}")
        else:
            say(f"  {BAD} port {port} already in use by {label}")
            problems.append(
                f"Port {port} is busy. Re-run with --kill to free it, or stop {label}."
            )


def preflight(python: str, args: argparse.Namespace) -> list[str]:
    problems: list[str] = []
    ports = [FASTAPI_PORT, EXPRESS_PORT] + ([FRONTEND_PORT] if not args.no_frontend else [])

    rule("Preflight")
    say(" Runtime")
    check_python(problems)
    check_py_packages(python, problems, args.install)
    check_node(problems, args.install, not args.no_frontend)

    say("\n Configuration")
    check_env_files()

    say("\n Model weights")
    check_weights()

    say("\n Decision-support artefacts")
    check_artefacts()

    say("\n Services")
    check_mongo()
    check_ports(ports, args.kill, problems)
    return problems


# ---------------------------------------------------------------------------
# start / stop
# ---------------------------------------------------------------------------

def build_services(python: str, want_frontend: bool) -> list[dict]:
    npm = find_exe("npm") or "npm"
    node = find_exe("node") or "node"
    services = [
        {
            "name": "FastAPI (ML service)",
            # Must run from backend/fastapi: model paths in config are relative
            # and resolve against the working directory.
            "cwd": FASTAPI_DIR,
            "cmd": [python, "-m", "uvicorn", "app.main:app",
                    "--host", "127.0.0.1", "--port", str(FASTAPI_PORT)],
            "health": f"http://127.0.0.1:{FASTAPI_PORT}/health",
            # Loads four checkpoints (~645 MB) on a cold start.
            "timeout": 180,
        },
        {
            "name": "Express (API gateway)",
            "cwd": EXPRESS_DIR,
            "cmd": [node, "src/server.js"],
            "health": f"http://127.0.0.1:{EXPRESS_PORT}/api/health",
            "timeout": 60,
        },
    ]
    if want_frontend:
        services.append({
            "name": "Frontend (Vite)",
            "cwd": FRONTEND_DIR,
            "cmd": [npm, "run", "dev"],
            "health": f"http://127.0.0.1:{FRONTEND_PORT}/",
            "timeout": 90,
        })
    return services


def start(services: list[dict]) -> list[dict]:
    rule("Starting services")
    running: list[dict] = []
    for svc in services:
        try:
            proc = subprocess.Popen(
                svc["cmd"], cwd=str(svc["cwd"]),
                creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP
                               if sys.platform == "win32" else 0),
            )
        except FileNotFoundError:
            say(f"  {BAD} {svc['name']}: {svc['cmd'][0]} not found")
            stop(running)
            sys.exit(1)
        except Exception as exc:
            say(f"  {BAD} {svc['name']}: {exc}")
            stop(running)
            sys.exit(1)
        running.append({**svc, "proc": proc})
        say(f"  {DOT} {svc['name']} launched (PID {proc.pid})")
    return running


def wait_ready(running: list[dict]) -> bool:
    say("\n  Waiting for services to answer their health endpoints...")
    all_ok = True
    for svc in running:
        deadline = time.time() + svc["timeout"]
        started = time.time()
        while time.time() < deadline:
            if svc["proc"].poll() is not None:
                say(f"  {BAD} {svc['name']} exited during startup "
                    f"(code {svc['proc'].returncode}). Its output is above.")
                all_ok = False
                break
            if http_ok(svc["health"]):
                say(f"  {OK} {svc['name']} ready in {time.time() - started:.1f}s")
                break
            time.sleep(1.0)
        else:
            say(f"  {WARN} {svc['name']} did not answer {svc['health']} "
                f"within {svc['timeout']}s - continuing anyway")
            all_ok = False
    return all_ok


def stop(running: list[dict]) -> None:
    if not running:
        return
    say("\n  Stopping services...")
    for svc in running:
        proc = svc["proc"]
        if proc.poll() is not None:
            continue
        try:
            # /T so npm's child (the actual Vite process) dies too; terminating
            # only the parent leaves an orphan holding port 3000.
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/F", "/T"],
                               capture_output=True, timeout=15)
            else:
                proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        say(f"  {OK} {svc['name']} stopped")
    say("\n  All services stopped.")


def print_urls(want_frontend: bool) -> None:
    rule("Ready")
    say("  Open this:")
    if want_frontend:
        say(f"    Dashboard      http://localhost:{FRONTEND_PORT}/dashboard.html")
        say(f"    Landing page   http://localhost:{FRONTEND_PORT}/")
        say(f"    Research       http://localhost:{FRONTEND_PORT}/research.html")
    say(f"    API docs       http://localhost:{FASTAPI_PORT}/docs")
    say(f"    Gateway health http://localhost:{EXPRESS_PORT}/api/health")
    say("")
    say("  Try a scan from the command line:")
    say(f"    curl -X POST \"http://localhost:{EXPRESS_PORT}/api/scan?module=brain_mri\" \\")
    say("         -F \"image=@frontend/samples/meningioma_sample.jpg\"")
    say("")
    say("  Note: the first brain scan after startup takes ~15s while the four")
    say("  models load. Subsequent scans are ~650ms.")
    say("")
    say("  Ctrl+C to stop everything.")
    rule()


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start the NeuraSight stack.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python run.py                 start everything\n"
               "  python run.py --check         preflight only\n"
               "  python run.py --install       install missing deps, then start\n"
               "  python run.py --kill          free the ports and exit\n"
               "  python run.py --no-frontend   API services only\n",
    )
    parser.add_argument("--check", action="store_true",
                        help="run the preflight checks and exit")
    parser.add_argument("--install", action="store_true",
                        help="install missing Python/Node dependencies")
    parser.add_argument("--kill", action="store_true",
                        help="free ports 8000/5000/3000 before starting")
    parser.add_argument("--no-frontend", action="store_true",
                        help="start FastAPI and Express only")
    args = parser.parse_args()

    python = find_python()
    rule("NeuraSight")
    say(f"  repo        {ROOT}")
    say(f"  interpreter {python}")

    problems = preflight(python, args)

    # `--kill` on its own is a maintenance action, not a launch.
    if sys.argv[1:] == ["--kill"]:
        rule("Ports freed")
        say("  Nothing started (--kill only). Run 'python run.py' to start.")
        rule()
        return

    if problems:
        rule("Cannot start yet")
        for i, p in enumerate(problems, 1):
            say(f"  {i}. {p}")
        say("")
        say("  Fix the above and re-run. 'python run.py --install' handles the")
        say("  dependency items automatically; '--kill' frees busy ports.")
        rule()
        sys.exit(1)

    if args.check:
        rule("Preflight passed")
        say("  Everything needed is in place. Run 'python run.py' to start.")
        rule()
        return

    running = start(build_services(python, not args.no_frontend))
    wait_ready(running)
    print_urls(not args.no_frontend)

    try:
        while True:
            dead = [s for s in running if s["proc"].poll() is not None]
            if dead:
                for svc in dead:
                    say(f"\n  {BAD} {svc['name']} exited unexpectedly "
                        f"(code {svc['proc'].returncode}). Its output is above.")
                say("\n  Shutting down the rest.")
                stop([s for s in running if s["proc"].poll() is None])
                sys.exit(1)
            time.sleep(2)
    except KeyboardInterrupt:
        stop(running)


if __name__ == "__main__":
    main()
