import subprocess,sys

def test_version():
    r=subprocess.run([sys.executable,'-m','mldoctor','--version'],capture_output=True,text=True)
    assert r.returncode==0 and '0.3.0' in r.stdout
