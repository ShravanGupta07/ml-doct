import gc
from rich.console import Console
from rich.panel import Panel
console=Console()

def tune_batch_size(model,dataset_sample_shape,max_batch=1024,min_batch=16):
    import torch
    if min_batch<1 or max_batch<min_batch: raise ValueError("Invalid batch-size bounds")
    try: device=next(model.parameters()).device
    except StopIteration: device=torch.device("cpu")
    low,high,optimal=min_batch,max_batch,min_batch
    console.print(Panel("Binary-searching a maximum batch size using real forward/backward passes.",title="ML Doctor Auto-Tuner",border_style="cyan"))
    while low<=high:
        mid=(low+high)//2
        try:
            model.train(); model.zero_grad(set_to_none=True)
            x=torch.randn(mid,*dataset_sample_shape,device=device)
            out=model(x); loss=out.sum(); loss.backward()
            model.zero_grad(set_to_none=True); del x,out,loss
            if device.type=="cuda": torch.cuda.empty_cache()
            gc.collect(); optimal=mid; low=mid+1
        except (RuntimeError,MemoryError) as e:
            if "out of memory" not in str(e).lower() and not isinstance(e,MemoryError): raise
            if device.type=="cuda": torch.cuda.empty_cache()
            gc.collect(); high=mid-1
    return optimal
