from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, Optional

class Severity(str, Enum):
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"

@dataclass
class Diagnostic:
    rule_id: str
    severity: Severity
    message: str
    fix: str = ""
    category: str = "correctness"
    path: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None
    confidence: float = 1.0
    fixable: bool = False
    evidence: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d
