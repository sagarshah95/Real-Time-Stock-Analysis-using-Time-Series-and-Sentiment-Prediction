"""Launch Streamlit without needing `streamlit` on PATH."""
import subprocess
import sys


def main():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"])
    subprocess.check_call([sys.executable, "-m", "streamlit", "run", "app.py"])


if __name__ == "__main__":
    main()
