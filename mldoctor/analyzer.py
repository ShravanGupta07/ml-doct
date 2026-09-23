from typing import List, Optional, Tuple
from .config import Config
from .linter import lint_source
from .models import Diagnostic, Severity

_ORDER={"INFO":0,"WARNING":1,"ERROR":2,"CRITICAL":3}

def analyze_file(path: str, config: Optional[Config]=None) -> Tuple[List[Diagnostic], object]:
    config=config or Config()
    with open(path,"r",encoding="utf-8") as f: source=f.read()
    findings, tree=lint_source(source,path)
    findings=[d for d in findings if config.allows(d.rule_id) and d.confidence>=config.confidence]
    return findings,tree

def has_failure(findings: List[Diagnostic], fail_on: str) -> bool:
    threshold=_ORDER.get(fail_on.upper(),2)
    return any(_ORDER.get(d.severity.value,2)>=threshold for d in findings)
