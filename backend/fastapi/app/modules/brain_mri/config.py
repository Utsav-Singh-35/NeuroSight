"""Brain MRI module configuration."""
import json
import os

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(MODULE_DIR, "metadata.json"), "r") as f:
    METADATA = json.load(f)
