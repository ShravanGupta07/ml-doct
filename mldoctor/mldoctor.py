import argparse,sys,json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from . import VERSION
from .analyzer import analyze_file,has_failure
from .config import load_config
from .fixer import apply_fixes
from .reporters import text_report,json_report,sarif_report
from .diagnose import diagnose_csv,generate_markdown_report
from .watchdog import run_watchdog
console=Console()

def show_info():
    console.print(Panel(f"[bold cyan]ML Doctor[/bold cyan] v{VERSION}\n\nStatic PyTorch analysis, safe fixes, runtime watchdog, and metric diagnosis.\nNo source execution occurs during `check`.",title="Package Information",border_style="cyan"))

def main():
    p=argparse.ArgumentParser(prog="mldoctor",description="ML Doctor: PyTorch training diagnostics")
    p.add_argument("-v","--version",action="version",version=f"mldoctor {VERSION}")
    sp=p.add_subparsers(dest="command")
    c=sp.add_parser("check",help="Statically analyze a Python training script")
    c.add_argument("file"); c.add_argument("--fix",action="store_true"); c.add_argument("--dry-run",action="store_true"); c.add_argument("--no-backup",action="store_true")
    c.add_argument("--format",choices=["text","json","sarif"],default="text"); c.add_argument("--output"); c.add_argument("--config")
    c.add_argument("--select",action="append",default=[]); c.add_argument("--ignore",action="append",default=[]); c.add_argument("--fail-on",choices=["INFO","WARNING","ERROR","CRITICAL"]); c.add_argument("--min-confidence",type=float)
    d=sp.add_parser("diagnose",help="Diagnose training metric CSV logs")
    d.add_argument("path"); d.add_argument("--run-id",type=int); d.add_argument("--export-report",action="store_true"); d.add_argument("--output",default="mldoctor_report.md"); d.add_argument("--model")
    w=sp.add_parser("watch",help="Run a training script under the runtime watchdog")
    w.add_argument("script"); w.add_argument("script_args",nargs=argparse.REMAINDER); w.add_argument("--patience",type=int,default=10)
    sp.add_parser("info")
    args=p.parse_args()
    if args.command in (None,"info"): show_info(); return 0
    if args.command=="check":
        cfg=load_config(args.config);
        if args.select: cfg.select=args.select
        if args.ignore: cfg.ignore=args.ignore
        if args.fail_on: cfg.fail_on=args.fail_on
        if args.min_confidence is not None: cfg.confidence=args.min_confidence
        try: findings,_=analyze_file(args.file,cfg)
        except (OSError,UnicodeError) as e: console.print(f"[red]Check failed: {e}[/red]"); return 2
        if args.fix and findings:
            ok,msg,diff=apply_fixes(args.file,findings,dry_run=args.dry_run,backup=not args.no_backup)
            console.print(f"[green]{msg}[/green]" if ok else f"[yellow]{msg}[/yellow]")
            if diff and args.dry_run: console.print(diff)
        report={"text":text_report,"json":json_report,"sarif":sarif_report}[args.format](findings)
        if args.output:
            with open(args.output,"w",encoding="utf-8") as f: f.write(report)
        else: print(report,end="")
        return 1 if has_failure(findings,cfg.fail_on) else 0
    if args.command=="diagnose":
        try: result=diagnose_csv(args.path,args.model,args.run_id)
        except Exception as e: console.print(f"[red]Diagnosis failed: {e}[/red]"); return 2
        console.print(Panel(f"[bold]Condition:[/bold] {result['title']}\n[bold]Confidence:[/bold] {result['confidence']:.2f}%\n\n{result['desc']}\n\n[bold green]Suggested checks:[/bold green] {result['action']}\n\n"+"\n".join("• "+x for x in result['insights']),title="ML Doctor Diagnosis",border_style="cyan"))
        if args.export_report: console.print(f"[green]Report: {generate_markdown_report(result,args.output)}[/green]")
        return 0
    if args.command=="watch": return run_watchdog(args.script,args.script_args,patience=args.patience)
    return 0

if __name__=="__main__": raise SystemExit(main())
