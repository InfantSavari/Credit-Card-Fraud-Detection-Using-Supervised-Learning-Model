from xgboost import XGBClassifier
import pandas as pd
import joblib

x_train = pd.read_parquet(r"Train_and_Test\x_train.parquet")
y_train = pd.read_parquet(r"Train_and_Test\y_train.parquet").values.ravel()

scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss"
)

model.fit(x_train, y_train)
joblib.dump(model,r"Models\Xboost.pkl")