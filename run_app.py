"""
Run Streamlit application
"""

import subprocess
import sys
from pathlib import Path


def main():
    # New location: src/web/streamlit_app.py
    app_path = Path(__file__).parent / "src" / "web" / "streamlit_app.py"
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
    ])


if __name__ == "__main__":
    main()
