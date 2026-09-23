import pandas as pd
from mldoctor.diagnose import extract_features_from_run

def test_feature_drop_matches_training_definition():
    df=pd.DataFrame({'train_loss':[10.0,5.0],'val_loss':[9.0,6.0],'grad_norm':[1.0,2.0]})
    X,_=extract_features_from_run(df)
    assert abs(X.iloc[0]['t_loss_drop']-0.5)<1e-9
