import pandas as pd
import glob

parquet_file = glob.glob("input/*.parquet")
data = pd.concat((pd.read_parquet(p) for p in parquet_file),ignore_index=True)
df = data.drop(columns=['nameOrig', 'nameDest'],inplace=True)

df['sender_balance_change'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['dest_balance_change'] = df['newbalanceDest'] - df['oldbalanceDest']
df = pd.get_dummies(df, columns=['type'], drop_first=True)
x = df.drop(columns=['isFraud'])
y = df['isFraud']

x.to_parquet(r"Preprocessed_data/featured_data.parquet")
y.to_frame().to_parquet(r"Preprocessed_data/fraud.parquet")
