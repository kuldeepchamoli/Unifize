import csv
from datetime import datetime, timezone
from pathlib import Path

_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "usage.csv"
_FIELDS = ["timestamp", "model", "input_tokens", "output_tokens"]


def log_usage(model: str, input_tokens: int, output_tokens: int) -> None:
    _LOG_DIR.mkdir(exist_ok=True)
    write_header = not _LOG_FILE.exists()
    with _LOG_FILE.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        })
