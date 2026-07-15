"""
NeuraSight - Start All Services
================================
Run this script to start the entire NeuraSight application:
  - FastAPI ML Service (port 8000)
  - Express Backend (port 5000)
  - Frontend Dev Server (port 3000)

Usage:
    python run.py

Press Ctrl+C to stop all services.
"""

import subprocess
import sys
import time
import os
import signal

# Project root
ROOT = os.path.dirname(os.path.abspath(__file__))

# Service configurations
SERVICES = [
    {
        "name": "FastAPI (ML Service)",
        "port": 8000,
        "cmd": ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"],
        "cwd": os.path.join(ROOT, "backend", "fastapi"),
    },
    {
        "name": "Express (Backend)",
        "port": 5000,
        "cmd": ["node", "src/server.js"],
        "cwd": os.path.join(ROOT, "backend", "express"),
    },
    {
        "name": "Frontend (Vite)",
        "port": 3000,
        "cmd": ["npm", "run", "dev"],
        "cwd": os.path.join(ROOT, "frontend"),
    },
]

processes = []


def start_services():
    """Start all services as subprocesses."""
    print("=" * 60)
    print("  NeuraSight - Starting All Services")
    print("=" * 60)
    print()

    for service in SERVICES:
        print(f"  Starting {service['name']}...")
        try:
            proc = subprocess.Popen(
                service["cmd"],
                cwd=service["cwd"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
            )
            processes.append({"proc": proc, **service})
            print(f"  ✓ {service['name']} started (PID: {proc.pid})")
        except Exception as e:
            print(f"  ✗ Failed to start {service['name']}: {e}")
            stop_services()
            sys.exit(1)

    # Wait for services to boot
    print()
    print("  Waiting for services to initialize...")
    time.sleep(5)

    print()
    print("=" * 60)
    print("  All services running!")
    print("=" * 60)
    print()
    print("  URLs:")
    print(f"    Frontend:     http://localhost:3000")
    print(f"    Dashboard:    http://localhost:3000/dashboard.html")
    print(f"    Express API:  http://localhost:5000/api/health")
    print(f"    FastAPI Docs: http://localhost:8000/docs")
    print()
    print("  Press Ctrl+C to stop all services.")
    print("=" * 60)


def stop_services():
    """Stop all running services."""
    print()
    print("  Stopping all services...")
    for entry in processes:
        proc = entry["proc"]
        name = entry["name"]
        if proc.poll() is None:
            try:
                if sys.platform == "win32":
                    proc.terminate()
                else:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=5)
                print(f"  ✓ {name} stopped")
            except Exception:
                proc.kill()
                print(f"  ✓ {name} killed")
    print()
    print("  All services stopped. Goodbye!")


def main():
    # Check prerequisites
    model_path = os.path.join(ROOT, "models", "Brain_MRI_scan.pth")
    if not os.path.exists(model_path):
        print("  ⚠ WARNING: Model file not found at models/Brain_MRI_scan.pth")
        print("    The FastAPI service will fail to start.")
        print("    Place your trained model file there or run:")
        print("      python models/create_dummy_model.py")
        print()

    express_modules = os.path.join(ROOT, "backend", "express", "node_modules")
    if not os.path.exists(express_modules):
        print("  ⚠ Express dependencies not installed. Running npm install...")
        subprocess.run(["npm", "install"], cwd=os.path.join(ROOT, "backend", "express"), shell=True)
        print()

    start_services()

    try:
        # Keep running until Ctrl+C
        while True:
            # Check if any process died
            for entry in processes:
                if entry["proc"].poll() is not None:
                    print(f"\n  ⚠ {entry['name']} has stopped unexpectedly!")
            time.sleep(2)
    except KeyboardInterrupt:
        stop_services()


if __name__ == "__main__":
    main()
