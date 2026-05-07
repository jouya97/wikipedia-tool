"""
Root conftest.py — ensures the repo root is on sys.path so pytest can import
all top-level packages (datagen, wiki, agent, eval) without a package install.
"""
import sys
import os

# Add the repo root to sys.path if not already present
ROOT = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
