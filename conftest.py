"""Make the repo root importable so tests can import audio_dataset.* and src.*."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Register the project's custom pytest marks. Without this, `@pytest.mark.slow`
# raises PytestUnknownMarkWarning on every run and looks like a typo in CI output.
def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers", "slow: disk-heavy tests (e.g. hashing 3,000 audio files)"
    )
