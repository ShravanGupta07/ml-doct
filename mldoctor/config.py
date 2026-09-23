import os
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib

@dataclass
class Config:
    select: List[str] = field(default_factory=list)
    ignore: List[str] = field(default_factory=list)
    fail_on: str = "ERROR"
    confidence: float = 0.0
    workers_warning: int = 0

    def allows(self, rule_id: str) -> bool:
        if self.select and rule_id not in self.select:
            return False
        return rule_id not in self.ignore

def load_config(path: Optional[str] = None) -> Config:
    candidates = [path] if path else []
    if not path:
        candidates.extend(["pyproject.toml", ".mldoctor.toml"])
    chosen = next((p for p in candidates if p and os.path.isfile(p)), None)
    if not chosen:
        return Config()
    with open(chosen, "rb") as f:
        data = tomllib.load(f)
    section = data.get("tool", {}).get("mldoct", {})
    return Config(
        select=list(section.get("select", [])),
        ignore=list(section.get("ignore", [])),
        fail_on=str(section.get("fail_on", "ERROR")).upper(),
        confidence=float(section.get("confidence", 0.0)),
        workers_warning=int(section.get("workers_warning", 0)),
    )
