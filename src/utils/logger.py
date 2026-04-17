from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable

from src.utils.io import ensure_dir


class CSVLogger:
    def __init__(self, path: str | Path, fieldnames: Iterable[str]) -> None:
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self.fieldnames = list(fieldnames)
        self._initialized = False

    def log(self, row: Dict[str, object]) -> None:
        write_header = not self.path.exists() or not self._initialized
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if write_header:
                writer.writeheader()
                self._initialized = True
            writer.writerow(row)
