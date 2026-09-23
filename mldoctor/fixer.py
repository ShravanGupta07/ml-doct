import difflib, os, shutil, tempfile
from typing import List, Tuple
from .models import Diagnostic

def _indent(line):
    return line[:len(line)-len(line.lstrip())]

def build_fixed_source(source: str, findings: List[Diagnostic]) -> Tuple[str,List[str]]:
    lines=source.splitlines(True)
    inserts=[]
    for d in findings:
        if not d.fixable or d.line is None:
            continue
        if d.rule_id == "MD001":
            line=lines[d.line-1]
            indent=_indent(line)
            opt="optimizer"
            import re
            m=re.search(r"(\w+)\.zero_grad", d.fix or "")
            if m:
                opt=m.group(1)
            inserts.append((d.line-1, f"{indent}{opt}.zero_grad()  # [ML Doctor MD001]\n"))
        elif d.rule_id == "MD002":
            line=lines[d.line-1]
            indent=_indent(line)
            inserts.append((d.line-1, f"{indent}model.eval()  # [ML Doctor MD002]\n"))
    for idx,text in sorted(inserts, reverse=True):
        lines.insert(idx,text)
    return "".join(lines), [f"MD001 inserted {text.strip()}" for _,text in inserts]

def apply_fixes(path: str, findings: List[Diagnostic], dry_run=False, backup=True) -> Tuple[bool,str,str]:
    with open(path,"r",encoding="utf-8") as f:
        original=f.read()
    fixed,changes=build_fixed_source(original,findings)
    if fixed==original:
        return False,"No safe automated fixes available.",""
    try:
        compile(fixed,path,"exec")
    except SyntaxError as e:
        return False,f"Refused to write fix because generated source is invalid: {e}",""
    diff="".join(difflib.unified_diff(original.splitlines(True),fixed.splitlines(True),fromfile=path,tofile=path+".fixed"))
    if dry_run:
        return True,"Dry run only; no file was changed.",diff
    backup_path=path+".bak"
    if backup:
        shutil.copy2(path,backup_path)
    directory=os.path.dirname(os.path.abspath(path)) or "."
    fd,tmp=tempfile.mkstemp(prefix=".mldoct-",suffix=".py",dir=directory)
    try:
        with os.fdopen(fd,"w",encoding="utf-8",newline="") as f:
            f.write(fixed)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return True,f"Applied safe fixes. Backup: {backup_path if backup else 'disabled'}",diff
