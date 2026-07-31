"""
NeuraSight — One-Click Setup
==============================
Run this script after cloning the repo to install all dependencies.

Usage:
    python setup.py

What it does:
    1. Creates a Python virtual environment (backend/fastapi/venv)
    2. Installs Python dependencies (PyTorch, FastAPI, timm, etc.)
    3. Installs Node.js dependencies for Express backend
    4. Installs Node.js dependencies for React frontend
    5. Copies .env.example → .env if .env doesn't exist
    6. Verifies model files exist in models/
    7. Checks if MongoDB is reachable

After setup:
    python run.py

Requirements:
    - Python 3.10+
    - Node.js 18+
    - MongoDB running on localhost:27017 (or configure .env)
    - Model weights in models/ (download from Google Drive)
"""

import subprocess
import sys
import os
import shutil
import platform

ROOT = os.path.dirname(os.path.abspath(__file__))

# Paths
FASTAPI_DIR = os.path.join(ROOT, "backend", "fastapi")
EXPRESS_DIR = os.path.join(ROOT, "backend", "express")
FRONTEND_DIR = os.path.join(ROOT, "frontend")
MODELS_DIR = os.path.join(ROOT, "models")
VENV_DIR = os.path.join(FASTAPI_DIR, "venv")


def print_header(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def print_step(msg):
    print(f"\n  → {msg}")


def print_ok(msg):
    print(f"  ✓ {msg}")


def print_warn(msg):
    print(f"  ⚠ {msg}")


def print_err(msg):
    print(f"  ✗ {msg}")


def run_cmd(cmd, cwd=None, shell=True):
    """Run a shell command, print output on failure."""
    result = subprocess.run(
        cmd, cwd=cwd, shell=shell,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print_err(f"Command failed: {cmd}")
        if result.stderr:
            print(f"    {result.stderr[:500]}")
        return False
    return True


def check_prerequisites():
    """Check Python and Node.js are available."""
    print_header("Checking Prerequisites")

    # Python
    py_version = sys.version_info
    if py_version >= (3, 10):
        print_ok(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    else:
        print_err(f"Python 3.10+ required, found {py_version.major}.{py_version.minor}")
        sys.exit(1)

    # Node.js
    try:
        result = subprocess.run(
            ["node", "--version"], capture_output=True, text=True, shell=True
        )
        node_ver = result.stdout.strip()
        major = int(node_ver.replace("v", "").split(".")[0])
        if major >= 18:
            print_ok(f"Node.js {node_ver}")
        else:
            print_err(f"Node.js 18+ required, found {node_ver}")
            sys.exit(1)
    except Exception:
        print_err("Node.js not found. Install from https://nodejs.org/")
        sys.exit(1)

    # npm
    try:
        result = subprocess.run(
            ["npm", "--version"], capture_output=True, text=True, shell=True
        )
        print_ok(f"npm {result.stdout.strip()}")
    except Exception:
        print_err("npm not found")
        sys.exit(1)


def setup_python_env():
    """Create venv and install Python dependencies."""
    print_header("Setting Up Python Environment")

    # Create virtual environment
    if not os.path.exists(VENV_DIR):
        print_step("Creating virtual environment...")
        run_cmd(f'"{sys.executable}" -m venv "{VENV_DIR}"')
        print_ok("Virtual environment created")
    else:
        print_ok("Virtual environment already exists")

    # Determine pip path
    if platform.system() == "Windows":
        pip = os.path.join(VENV_DIR, "Scripts", "pip.exe")
        python = os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:
        pip = os.path.join(VENV_DIR, "bin", "pip")
        python = os.path.join(VENV_DIR, "bin", "python")

    # Upgrade pip
    print_step("Upgrading pip...")
    run_cmd(f'"{python}" -m pip install --upgrade pip', cwd=FASTAPI_DIR)

    # Install requirements
    print_step("Installing Python dependencies (this may take a few minutes)...")
    req_file = os.path.join(FASTAPI_DIR, "requirements.txt")
    success = run_cmd(f'"{pip}" install -r "{req_file}"', cwd=FASTAPI_DIR)
    if success:
        print_ok("Python dependencies installed")
    else:
        print_err("Failed to install Python dependencies")
        print("    Try manually: cd backend/fastapi && pip install -r requirements.txt")


def setup_express():
    """Install Express backend dependencies."""
    print_header("Setting Up Express Backend")

    if os.path.exists(os.path.join(EXPRESS_DIR, "node_modules")):
        print_ok("node_modules already exists")
    else:
        print_step("Installing Express dependencies...")
        success = run_cmd("npm install", cwd=EXPRESS_DIR)
        if success:
            print_ok("Express dependencies installed")
        else:
            print_err("Failed to install Express dependencies")


def setup_frontend():
    """Install frontend dependencies."""
    print_header("Setting Up Frontend")

    if os.path.exists(os.path.join(FRONTEND_DIR, "node_modules")):
        print_ok("node_modules already exists")
    else:
        print_step("Installing frontend dependencies...")
        success = run_cmd("npm install", cwd=FRONTEND_DIR)
        if success:
            print_ok("Frontend dependencies installed")
        else:
            print_err("Failed to install frontend dependencies")


def setup_env_files():
    """Copy .env.example → .env if .env doesn't exist."""
    print_header("Setting Up Environment Files")

    env_pairs = [
        (os.path.join(FASTAPI_DIR, ".env.example"), os.path.join(FASTAPI_DIR, ".env")),
        (os.path.join(EXPRESS_DIR, ".env.example"), os.path.join(EXPRESS_DIR, ".env")),
    ]

    for example, target in env_pairs:
        if os.path.exists(target):
            print_ok(f"{os.path.relpath(target, ROOT)} already exists")
        elif os.path.exists(example):
            shutil.copy2(example, target)
            print_ok(f"Created {os.path.relpath(target, ROOT)} from .env.example")
        else:
            print_warn(f"No .env.example found at {os.path.relpath(example, ROOT)}")


def check_models():
    """Verify model weight files exist."""
    print_header("Checking Model Files")

    required_models = [
        "BRAIN_MRI_EFFICIENTNET.pth",
        "BRAIN_MRI_RESNET.pth",
        "BRAIN_MRI_DENSENET.pth",
        "BRAIN_MRI_VGG.pth",
        "meta_model.pkl",
        "ensemble_config.json",
    ]

    all_found = True
    for model_file in required_models:
        path = os.path.join(MODELS_DIR, model_file)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print_ok(f"{model_file} ({size_mb:.1f} MB)")
        else:
            print_warn(f"{model_file} NOT FOUND")
            all_found = False

    if not all_found:
        print()
        print("  Some model files are missing. The ensemble needs all of them.")
        print("  Download from Google Drive: My Drive/NeuraSight/models/")
        print()
        print("  Alternatively, run with single-model mode:")
        print("    Set USE_ENSEMBLE=False in backend/fastapi/.env")
        print("    (only BRAIN_MRI_EFFICIENTNET.pth is required)")


def check_mongodb():
    """Check if MongoDB is reachable."""
    print_header("Checking MongoDB")

    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex(("localhost", 27017))
        sock.close()
        if result == 0:
            print_ok("MongoDB is running on localhost:27017")
        else:
            print_warn("MongoDB not reachable on localhost:27017")
            print("    Start MongoDB before running the application.")
            print("    Windows: net start MongoDB")
            print("    macOS:   brew services start mongodb-community")
            print("    Linux:   sudo systemctl start mongod")
    except Exception:
        print_warn("Could not check MongoDB status")


def print_summary():
    """Print final setup summary."""
    print_header("Setup Complete!")
    print()
    print("  To start NeuraSight:")
    print()
    print("    python run.py")
    print()
    print("  This starts all 3 services:")
    print("    • FastAPI ML Service  → http://localhost:8000")
    print("    • Express API Gateway → http://localhost:5000")
    print("    • React Frontend      → http://localhost:3000")
    print()
    print("  Open http://localhost:3000/dashboard.html in your browser")
    print()
    print("  Notes:")
    print("    • Ensure MongoDB is running before starting")
    print("    • Model files must be in the models/ folder")
    print("    • If you get memory errors (8GB RAM), set USE_ENSEMBLE=False")
    print("      in backend/fastapi/.env to use single-model mode (~100MB)")
    print()
    print(f"{'='*60}")


def main():
    print()
    print("  ╔══════════════════════════════════════╗")
    print("  ║   NeuraSight — One-Click Setup       ║")
    print("  ║   Brain Tumor Detection Platform     ║")
    print("  ╚══════════════════════════════════════╝")

    check_prerequisites()
    setup_python_env()
    setup_express()
    setup_frontend()
    setup_env_files()
    check_models()
    check_mongodb()
    print_summary()


if __name__ == "__main__":
    main()
