#!/usr/bin/env python3
"""
Launcher script for the Streamlit web application
"""
import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit application"""
    script_dir = Path(__file__).parent
    streamlit_app = script_dir / "apps" / "streamlit" / "streamlit_app.py"
    
    if not streamlit_app.exists():
        print(f"Error: Streamlit app not found at {streamlit_app}")
        sys.exit(1)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(streamlit_app)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStreamlit app stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()
