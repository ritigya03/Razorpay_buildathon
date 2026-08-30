"""Point the backend at a throwaway DB + fast replay before anything imports it."""
import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="sentinel-test-")
os.environ.setdefault("SENTINEL_DB_URL", f"sqlite:///{_TMP}/test.db")
os.environ.setdefault("SENTINEL_REPLAY_DAYS_PER_SEC", "20")
os.environ.setdefault("SENTINEL_REPLAY_AUTOSTART", "true")
