import sys

from app.cli import main


if __name__ == "__main__":
    if sys.version_info[:2] != (3, 11):
        raise SystemExit("This application requires Python 3.11.x")
    main()