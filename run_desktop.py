#!/usr/bin/env python3
"""
Launcher script for the desktop Tkinter application
"""
import subprocess
import sys
from pathlib import Path

def main():
    """Launch the desktop application"""
    script_dir = Path(__file__).parent
    desktop_app = script_dir / "apps" / "desktop" / "tkinter_app.py"
    
    if not desktop_app.exists():
        print(f"Error: Desktop app not found at {desktop_app}")
        sys.exit(1)
    
    try:
        subprocess.run([sys.executable, str(desktop_app)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running desktop app: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDesktop app stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()
