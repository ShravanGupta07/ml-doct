"""Backward-compatible linter API built on the Phase 1 rule engine."""
from typing import List, Tuple
from .models import Diagnostic
from .rules import analyze_tree
from .config import Config
import ast
import os

class AdvancedMLVisitor(ast.NodeVisitor):
    """Compatibility facade exposing the legacy visitor flags."""
    def __init__(self):
        self.issues=[]
        self.has_backward=False
        self.has_zero_grad=False
        self.has_eval=False
        self.backward_line=None
        self.no_grad_line=None
        self.uses_softmax=False
        self.uses_cross_entropy=False
        self.has_clip_grad=False

    def visit_Call(self,node):
        name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ""
        if name=="backward": self.has_backward=True; self.backward_line=self.backward_line or node.lineno
        elif name=="zero_grad": self.has_zero_grad=True
        elif name=="eval": self.has_eval=True
        elif name=="softmax": self.uses_softmax=True
        elif name in ("CrossEntropyLoss","cross_entropy"): self.uses_cross_entropy=True
        elif name in ("clip_grad_norm_","clip_grad_value_"): self.has_clip_grad=True
        self.generic_visit(node)

    def visit_With(self,node):
        for item in node.items:
            if isinstance(item.context_expr,ast.Call):
                fn=item.context_expr.func
                if isinstance(fn,ast.Attribute) and fn.attr=="no_grad": self.no_grad_line=self.no_grad_line or node.lineno
        self.generic_visit(node)

def lint_source(source: str, path="<string>") -> Tuple[List[Diagnostic], object]:
    try:
        tree=ast.parse(source,filename=path)
    except SyntaxError as e:
        return [Diagnostic("MD000","ERROR",f"Syntax error: {e.msg}","Fix the Python syntax before analysis.","syntax",path,e.lineno,e.offset,1.0,False)],None
    visitor=AdvancedMLVisitor(); visitor.visit(tree)
    return analyze_tree(tree,path),tree

def lint_script(file_path: str):
    if not os.path.exists(file_path): return [],None
    with open(file_path,"r",encoding="utf-8") as f: source=f.read()
    findings,tree=lint_source(source,file_path)
    visitor=AdvancedMLVisitor()
    if tree is not None: visitor.visit(tree)
    legacy=[]
    for d in findings:
        legacy.append({"severity":d.severity.value if hasattr(d.severity,"value") else str(d.severity),"message":d.message,"fix":d.fix,"type":d.rule_id,"line":d.line})
    return legacy,visitor

def auto_fix_script(file_path, visitor):
    from .fixer import apply_fixes
    findings,tree=lint_source(open(file_path,encoding="utf-8").read(),file_path)
    return apply_fixes(file_path,findings,dry_run=False,backup=True)[:2]
