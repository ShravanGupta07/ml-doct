import os, glob, math, pickle, warnings
import numpy as np
import pandas as pd

REQUIRED=("train_loss","val_loss","grad_norm")
PRESCRIPTIONS={
"healthy":{"title":"Stable training dynamics","desc":"The observed metrics show a broadly stable training trajectory.","action":"Continue monitoring validation metrics and checkpointing."},
"lr_too_high":{"title":"Possible learning-rate instability","desc":"The model dynamics are consistent with an overly aggressive learning rate.","action":"Review the learning rate and consider a smaller value or a scheduler."},
"lr_too_low":{"title":"Possible slow optimization","desc":"The trajectory is consistent with very small parameter updates.","action":"Review learning rate, normalization, and optimizer settings."},
"overfitting":{"title":"Possible overfitting","desc":"Training and validation trajectories show a widening generalization gap.","action":"Consider early stopping, regularization, augmentation, or model-capacity changes."},
"underfitting":{"title":"Possible underfitting","desc":"Training loss remains relatively high with limited progress.","action":"Review model capacity, features, regularization, and optimization settings."},
"exploding_gradients":{"title":"Possible exploding gradients","desc":"Gradient magnitude or loss shows unstable growth.","action":"Review learning rate, initialization, normalization, and gradient clipping."},
"vanishing_gradients":{"title":"Possible vanishing gradients","desc":"Gradient magnitude is unusually small across the observed run.","action":"Review depth, activations, initialization, normalization, and residual paths."},
"missing_zero_grad":{"title":"Possible gradient accumulation","desc":"The trajectory is associated with a run labeled as missing gradient reset in the training dataset.","action":"Verify optimizer.zero_grad() placement in the training loop."},
"label_noise":{"title":"Possible label/data noise","desc":"The trajectory is consistent with noisy or contradictory supervision.","action":"Audit labels, preprocessing, class mappings, and per-sample losses."},
"high_variance_batch":{"title":"High training variance","desc":"The trajectory shows substantial short-term variation.","action":"Review batch size, learning rate, data ordering, and augmentation."},
}
FEATURES=["t_loss_start","t_loss_end","v_loss_start","v_loss_end","t_loss_drop","overfit_gap","t_loss_var","grad_max","grad_mean","grad_min","has_collapsed"]

def resolve_metric_file(path):
    if os.path.isfile(path): return path
    if os.path.isdir(path):
        files=glob.glob(os.path.join(path,"*.csv"))
        if not files: raise FileNotFoundError("No CSV metric logs found in directory '%s'."%path)
        return max(files,key=os.path.getmtime)
    raise FileNotFoundError("Path '%s' does not exist."%path)

def extract_features_from_run(df):
    missing=[c for c in REQUIRED if c not in df.columns]
    if missing: raise ValueError("Missing required metric columns: "+", ".join(missing))
    d=df.copy()
    for c in REQUIRED: d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.replace([np.inf,-np.inf],np.nan).dropna(subset=list(REQUIRED))
    if len(d)<2: raise ValueError("At least two valid metric rows are required.")
    t=d.train_loss.to_numpy(float); v=d.val_loss.to_numpy(float); g=d.grad_norm.to_numpy(float)
    start=float(t[0]); end=float(t[-1]);
    row={"t_loss_start":start,"t_loss_end":end,"v_loss_start":float(v[0]),"v_loss_end":float(v[-1]),
         "t_loss_drop":float((start-end)/(abs(start)+1e-8)),"overfit_gap":float(v[-1]-t[-1]),
         "t_loss_var":float(np.var(t)),"grad_max":float(np.max(g)),"grad_mean":float(np.mean(g)),"grad_min":float(np.min(g)),
         "has_collapsed":float(np.any(~np.isfinite(t)) or np.any(~np.isfinite(g)) or end>=99.0 or np.max(g)>=999.0)}
    return pd.DataFrame([row],columns=FEATURES),d

def _heuristic_label(d):
    t=d.train_loss.to_numpy(float); v=d.val_loss.to_numpy(float); g=d.grad_norm.to_numpy(float)
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(v)) or not np.all(np.isfinite(g)) or np.max(g)>=999 or np.max(t)>=99: return "exploding_gradients",0.98
    if np.mean(g)<1e-4: return "vanishing_gradients",0.94
    if len(v)>=3 and np.argmin(v)<len(v)-1 and v[-1]>np.min(v)*1.15: return "overfitting",0.86
    if len(t)>=5 and np.std(t[-5:])>max(0.2,0.1*max(abs(np.mean(t[-5:])),1e-8)): return "high_variance_batch",0.72
    drop=(t[0]-t[-1])/(abs(t[0])+1e-8)
    if drop<0.02: return "lr_too_low",0.68
    return "healthy",0.62

def diagnose_csv(target_path, model_path=None, run_id=None):
    csv_file=resolve_metric_file(target_path); df=pd.read_csv(csv_file)
    if "run_id" in df.columns:
        target_id=run_id if run_id is not None else df["run_id"].iloc[0]
        df=df[df["run_id"]==target_id]
        if df.empty: raise ValueError("run_id %r was not found in %s"%(run_id,csv_file))
    X,clean=extract_features_from_run(df)
    prediction=None; confidence=None; model_error=None
    heuristic, heuristic_conf = _heuristic_label(clean)
    strong_heuristic = heuristic_conf >= 0.84
    if strong_heuristic:
        prediction = heuristic
        confidence = heuristic_conf * 100
    if model_path is None: model_path=os.path.join(os.path.dirname(__file__),"doctor_model.pkl")
    if prediction is None and os.path.exists(model_path):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with open(model_path,"rb") as f: model=pickle.load(f)
            prediction=str(model.predict(X)[0]); confidence=float(np.max(model.predict_proba(X)[0])*100)
        except Exception as e: model_error=str(e)
    if prediction is None:
        prediction,conf=_heuristic_label(clean); confidence=conf*100
    info=PRESCRIPTIONS.get(prediction,{"title":prediction,"desc":"Anomalous training dynamics detected.","action":"Inspect the metric curves and training configuration."})
    insights=generate_data_driven_insights(clean)
    return {"source_file":csv_file,"label":prediction,"confidence":confidence,"title":info["title"],"desc":info["desc"],"action":info["action"],"insights":insights,"model_error":model_error}

def generate_data_driven_insights(df):
    t=df.train_loss.to_numpy(float); v=df.val_loss.to_numpy(float); g=df.grad_norm.to_numpy(float); insights=[]
    mean=float(np.mean(g))
    if mean<0.01: insights.append("Vanishing-gradient signal: mean gradient norm is %.4g."%mean)
    elif mean>10: insights.append("Large-gradient signal: mean gradient norm is %.4g."%mean)
    else: insights.append("Gradient health: mean gradient norm is %.4g."%mean)
    best=int(np.argmin(v)); bestv=float(v[best]);
    if best<len(v)-1 and bestv>0 and v[-1]>bestv*1.15:
        insights.append("Validation loss increased %.1f%% from its best observed value at row %d."%(((v[-1]-bestv)/bestv)*100,best+1))
    if len(t)>=5:
        vol=float(np.std(t[-5:]));
        if vol>0.2: insights.append("Late training loss volatility is %.4f across the last five rows."%vol)
    return insights

def generate_markdown_report(result,output_path="mldoctor_report.md"):
    lines=["# ML Doctor Diagnostic Report","",f"- Source: `{result['source_file']}`",f"- Diagnosis: **{result['title']}**",f"- Model confidence: **{result['confidence']:.2f}%**","", "## Observed dynamics",result["desc"],"","## Suggested checks",result["action"],"","## Data-driven observations"]
    lines += [f"- {x}" for x in result.get("insights",[]) ] or ["- No additional observations."]
    if result.get("model_error"): lines += ["","> Embedded model was unavailable; heuristic fallback was used."]
    with open(output_path,"w",encoding="utf-8") as f: f.write("\n".join(lines)+"\n")
    return output_path
