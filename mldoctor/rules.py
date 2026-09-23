import ast
from typing import Dict, List, Optional, Tuple
from .models import Diagnostic, Severity

def _name(expr):
    if isinstance(expr, ast.Name): return expr.id
    if isinstance(expr, ast.Attribute): return expr.attr
    return ""

def _call_name(call):
    return _name(call.func) if isinstance(call, ast.Call) else ""

def _optimizer_names(tree):
    names=set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
            fn=_call_name(n.value).lower()
            if fn in {"adam","adamw","sgd","rmsprop","adagrad","adadelta","adamax","nadam","radam","lbfgs"} or "optimizer" in fn:
                for t in n.targets:
                    if isinstance(t, ast.Name): names.add(t.id)
    return names

def analyze_tree(tree, path="<string>") -> List[Diagnostic]:
    out=[]; optimizer_names=_optimizer_names(tree)
    backward=[]; zero=[]; no_grad=[]; eval_calls=[]; softmax=[]; ce=[]; clip=[]
    scheduler_steps=[]; optimizer_steps=[]
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            name=_call_name(n)
            low=name.lower()
            if low=="backward": backward.append(n)
            elif low=="zero_grad": zero.append(n)
            elif low=="eval": eval_calls.append(n)
            elif low=="softmax": softmax.append(n)
            elif low in {"crossentropyloss","nllloss"}: ce.append(n)
            elif low in {"clip_grad_norm_","clip_grad_value_"}: clip.append(n)
            elif low=="step":
                owner=n.func.value.id if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) else ""
                if owner in optimizer_names: optimizer_steps.append(n)
                if owner and ("sched" in owner.lower() or "scheduler" in owner.lower()): scheduler_steps.append(n)
            if low=="dataloader":
                for kw in n.keywords:
                    if kw.arg=="num_workers" and isinstance(kw.value,ast.Constant) and kw.value.value==0:
                        out.append(Diagnostic("MD008",Severity.WARNING,"DataLoader uses num_workers=0; this can limit input throughput.","Consider tuning num_workers for the target machine.","performance",path,n.lineno,n.col_offset,0.90,False,"num_workers=0"))
        elif isinstance(n,ast.With):
            for item in n.items:
                if isinstance(item.context_expr,ast.Call) and _call_name(item.context_expr)=="no_grad": no_grad.append(n)
        elif isinstance(n,ast.AugAssign) and isinstance(n.op,ast.Add) and isinstance(n.value,ast.Name) and "loss" in n.value.id.lower():
            out.append(Diagnostic("MD004",Severity.CRITICAL,"Raw loss tensor is accumulated with += and may retain its autograd graph.",f"Use '{n.value.id}.item()' (or detach the tensor) when accumulating scalar metrics.","memory",path,n.lineno,n.col_offset,0.97,True,f"{ast.unparse(n.value) if hasattr(ast,'unparse') else n.value.id}"))
    if backward and not zero:
        fixable=bool(optimizer_names)
        opt=sorted(optimizer_names)[0] if optimizer_names else "optimizer"
        out.append(Diagnostic("MD001",Severity.CRITICAL,"loss.backward() is present but no optimizer.zero_grad() call was found.",f"Call {opt}.zero_grad() once per training iteration before backward/forward work.","correctness",path,backward[0].lineno,backward[0].col_offset,0.94,fixable,"backward without zero_grad"))
    has_model_name=any(isinstance(n,ast.Name) and n.id=="model" for n in ast.walk(tree))
    if no_grad and not eval_calls:
        out.append(Diagnostic("MD002",Severity.WARNING,"torch.no_grad() is used without a visible model.eval() call.","Call model.eval() before validation/inference when Dropout/BatchNorm evaluation behavior is required.","correctness",path,no_grad[0].lineno,no_grad[0].col_offset,0.88,has_model_name,"no_grad without eval"))
    if softmax and ce:
        out.append(Diagnostic("MD003",Severity.WARNING,"Manual softmax and CrossEntropyLoss are both present; verify that logits are not softmaxed twice.","CrossEntropyLoss expects logits, so remove an explicit softmax before the loss when applicable.","correctness",path,softmax[0].lineno,softmax[0].col_offset,0.78,False,"softmax + CrossEntropyLoss"))
    if optimizer_steps and scheduler_steps:
        first_opt=min(x.lineno for x in optimizer_steps); first_sched=min(x.lineno for x in scheduler_steps)
        if first_sched < first_opt:
            out.append(Diagnostic("MD006",Severity.WARNING,"A scheduler.step() appears before optimizer.step().","For common PyTorch schedulers, call optimizer.step() before scheduler.step(). Follow the scheduler's documented contract.","correctness",path,first_sched,0,0.86,False,"scheduler step precedes optimizer step"))
    if backward and not clip:
        out.append(Diagnostic("MD007",Severity.INFO,"No gradient clipping call was detected.","Consider clip_grad_norm_() when the model/training dynamics are susceptible to exploding gradients; clipping is not universally required.","stability",path,None,None,0.55,False,"backward present; no clipping"))
    return out

def lint_source(source: str, path="<string>") -> Tuple[List[Diagnostic], Optional[ast.AST]]:
    try:
        tree=ast.parse(source,filename=path)
    except SyntaxError as e:
        return [Diagnostic("MD000",Severity.ERROR,f"Syntax error: {e.msg}","Fix the Python syntax before ML Doctor can analyze the file.","syntax",path,e.lineno,e.offset,1.0,False)],None
    return analyze_tree(tree,path),tree
