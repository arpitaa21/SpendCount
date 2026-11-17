"""Leakage-aware chronological comparison of daily spending forecast baselines."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def compare_models(daily_spending):
    y=np.asarray(daily_spending,dtype=float)
    if len(y)<30:return {"status":"insufficient_history","message":"At least 30 daily observations are recommended for a chronological holdout."}
    split=int(len(y)*.8); train,test=y[:split],y[split:]
    # One-step lag features; no future observations cross the chronological split.
    def features(a,i): return [a[max(0,i-7):i].mean(),a[max(0,i-28):i].mean(),a[max(0,i-7):i].std()]
    X=np.array([features(y,i) for i in range(7,len(y))]); targets=y[7:]
    offset=split-7; model=RandomForestRegressor(n_estimators=100,max_depth=4,random_state=23).fit(X[:offset],targets[:offset])
    pred_rf=model.predict(X[offset:]); pred_ma=np.array([y[max(0,i-7):i].mean() for i in range(split,len(y))])
    truth=y[split:]
    return {"split":"chronological 80/20","train_days":split,"test_days":len(truth),"random_forest":{"mae":round(mean_absolute_error(truth,pred_rf),2),"rmse":round(mean_squared_error(truth,pred_rf)**.5,2)},"moving_average":{"mae":round(mean_absolute_error(truth,pred_ma),2),"rmse":round(mean_squared_error(truth,pred_ma)**.5,2)},"note":"Synthetic/demo results are not evidence of real-world performance."}
