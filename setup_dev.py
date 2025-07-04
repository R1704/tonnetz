#!/usr/bin/env python3
"""
Development setup script for the Tonnetz library.

This script helps set up the development environment and run basic tests.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n🔧 {description}...")
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def main():
    """Main setup function."""
    print("🎵 Tonnetz Development Setup")
    print("============================")

    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Please run this script from the tonnetz project root directory")
        sys.exit(1)

    # Install in development mode
    if not run_command("pip install -e .", "Installing Tonnetz in development mode"):
        sys.exit(1)

    # Install development dependencies
    if not run_command(
        "pip install pytest pytest-cov black flake8 mypy jupyter",
        "Installing development dependencies",
    ):
        print("⚠️  Development dependencies installation failed, but continuing...")

    # Run basic tests
    if Path("tests").exists():
        run_command("python -m pytest tests/ -v", "Running basic tests")
    else:
        print("⚠️  No tests directory found, skipping tests")

    # Test CLI commands
    print("\n🔧 Testing CLI commands...")
    cli_tests = [
        ("tonnetz-simulate --help", "CLI simulate help"),
        ("tonnetz-visualize --help", "CLI visualize help"),
        (
            "python -c 'import tonnetz; print(tonnetz.__version__)'",
            "Python import test",
        ),
    ]

    for cmd, desc in cli_tests:
        run_command(cmd, desc)

    print("\n✨ Setup completed!")
    print("\nNext steps:")
    print("  1. Explore the examples/ directory")
    print("  2. Run: jupyter notebook examples/tonnetz_demo.ipynb")
    print("  3. Try: tonnetz-simulate --help")
    print("  4. Check out the README.md for more information")


if __name__ == "__main__":
    main()
