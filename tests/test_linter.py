from mldoctor.linter import lint_source

def test_missing_zero_grad():
    findings,_=lint_source("import torch\noptimizer = torch.optim.Adam(model.parameters())\nloss.backward()\n")
    assert any(x.rule_id=="MD001" for x in findings)

def test_loss_accumulation():
    findings,_=lint_source("total_loss += loss\n")
    assert any(x.rule_id=="MD004" for x in findings)

def test_no_execution():
    findings,_=lint_source("raise RuntimeError('must not execute')\n")
    assert findings==[]
