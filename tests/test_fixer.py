from pathlib import Path
from mldoctor.linter import lint_source
from mldoctor.fixer import apply_fixes

def test_safe_fix_creates_backup(tmp_path):
    p=tmp_path/'train.py'; p.write_text("import torch\noptimizer = torch.optim.Adam(model.parameters())\nloss.backward()\n")
    findings,_=lint_source(p.read_text(),str(p))
    ok,msg,diff=apply_fixes(str(p),findings)
    assert ok and (tmp_path/'train.py.bak').exists()
    assert 'zero_grad' in p.read_text()
    compile(p.read_text(),str(p),'exec')

def test_safe_eval_fix(tmp_path):
    p=tmp_path/'validate.py'
    p.write_text("import torch\nmodel = make_model()\nwith torch.no_grad():\n    y = model(x)\n")
    findings,_=lint_source(p.read_text(),str(p))
    assert any(x.rule_id=='MD002' and x.fixable for x in findings)
    ok,_,_=apply_fixes(str(p),findings)
    assert ok and 'model.eval()' in p.read_text()
    compile(p.read_text(),str(p),'exec')
