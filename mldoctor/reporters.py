import json
from typing import Iterable
from .models import Diagnostic

def text_report(findings: Iterable[Diagnostic]) -> str:
    findings = list(findings)
    if not findings:
        return "ML Doctor: no findings.\n"
    out = []
    for d in findings:
        loc = f"{d.path}:{d.line}" if d.path and d.line else (d.path or "")
        out.append(f"[{d.severity.value}] {d.rule_id} {loc} — {d.message} (confidence={d.confidence:.0%})")
        if d.fix:
            out.append(f"  Fix: {d.fix}")
    return "\n".join(out) + "\n"

def json_report(findings: Iterable[Diagnostic]) -> str:
    return json.dumps({"schema_version": "1.0", "findings": [d.to_dict() for d in findings]}, indent=2)

def sarif_report(findings: Iterable[Diagnostic]) -> str:
    rules = {}
    results = []
    levels = {"CRITICAL": "error", "ERROR": "error", "WARNING": "warning", "INFO": "note"}
    for d in findings:
        rules.setdefault(d.rule_id, {
            "id": d.rule_id,
            "name": d.rule_id,
            "shortDescription": {"text": d.message},
        })
        result = {
            "ruleId": d.rule_id,
            "level": levels[d.severity.value],
            "message": {"text": d.message},
        }
        if d.path and d.line:
            result["locations"] = [{"physicalLocation": {
                "artifactLocation": {"uri": d.path},
                "region": {"startLine": d.line, "startColumn": (d.column or 0) + 1},
            }}]
        results.append(result)
    payload = {
        "version": "2.1.0",
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "runs": [{
            "tool": {"driver": {
                "name": "ML Doctor",
                "version": "0.3.0",
                "rules": list(rules.values()),
            }},
            "results": results,
        }],
    }
    return json.dumps(payload, indent=2)
