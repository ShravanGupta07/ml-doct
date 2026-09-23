from mldoctor.models import Diagnostic, Severity
from mldoctor.reporters import json_report,sarif_report

def test_reports():
    d=Diagnostic('MD001',Severity.CRITICAL,'bad',path='x.py',line=3)
    assert 'MD001' in json_report([d])
    assert '2.1.0' in sarif_report([d])
