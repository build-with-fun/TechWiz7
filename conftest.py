"""Make the repo root importable so tests can import audio_dataset.* and src.*."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Register custom marks so pytest doesn't warn about them.
def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers", "slow: disk-heavy tests (e.g. hashing 3,000 audio files)"
    )
