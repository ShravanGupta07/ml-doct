import re, subprocess, sys, threading, time, os
from rich.console import Console
from rich.panel import Panel
console=Console()
NAN_INF_PATTERN=re.compile(r"\b(?:nan|inf|-nan|-inf)\b",re.I)
LOSS_VALUE_PATTERN=re.compile(r"(?:loss[:=\s]+)([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",re.I)
VAL_LOSS_PATTERN=re.compile(r"(?:val_loss|validation loss|val loss)[:=\s]+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",re.I)
class GPUProfiler:
    def __init__(self): self.vram_history=[]; self.monitoring=True
    def track_vram(self,process):
        while self.monitoring and process.poll() is None:
            try:
                r=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,nounits,noheader"],text=True,stderr=subprocess.DEVNULL,timeout=2)
                vals=[int(x.strip()) for x in r.splitlines() if x.strip()];
                if vals: self.vram_history.append(max(vals))
                if len(self.vram_history)>=10 and all(a<b for a,b in zip(self.vram_history[-10:-1],self.vram_history[-9:])):
                    console.print(Panel("GPU memory has increased at every recent sample; terminating the child process to reduce OOM risk.",title="ML Doctor VRAM Alert",border_style="red")); process.terminate(); self.monitoring=False
            except Exception: pass
            time.sleep(2)

def run_watchdog(script_path,extra_args=None,patience=10,plateau_delta=0.001):
    extra_args=extra_args or []
    if not os.path.isfile(script_path):
        console.print(f"[red]Training script not found: {script_path}[/red]"); return 2
    cmd=[sys.executable,"-u",script_path]+extra_args
    console.print(Panel("Launching: "+" ".join(cmd),title="ML Doctor Watchdog",border_style="cyan"))
    process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    profiler=GPUProfiler(); threading.Thread(target=profiler.track_vram,args=(process,),daemon=True).start()
    best=float("inf"); stale=0; forced=None
    try:
        for line in iter(process.stdout.readline,""):
            sys.stdout.write(line); sys.stdout.flush()
            if NAN_INF_PATTERN.search(line):
                console.print(Panel("NaN/Inf detected in child output. Training process will be terminated.",title="ML Doctor Stability Alert",border_style="red")); process.terminate(); forced=1; break
            m=LOSS_VALUE_PATTERN.search(line)
            if m:
                try:
                    if float(m.group(1))>10000: console.print("[yellow]Large loss value observed (>10000).[/yellow]")
                except ValueError: pass
            m=VAL_LOSS_PATTERN.search(line)
            if m:
                try:
                    val=float(m.group(1))
                    if val < best-plateau_delta: best=val; stale=0
                    else: stale+=1
                    if stale>=patience:
                        console.print(Panel(f"Validation loss did not improve for {patience} checks (best={best:.6g}).",title="ML Doctor Early Stop",border_style="yellow")); process.terminate(); forced=0; break
                except ValueError: pass
    except KeyboardInterrupt:
        process.terminate(); forced=130
    finally:
        profiler.monitoring=False
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired: process.kill(); process.wait()
    return process.returncode if forced is None else forced
