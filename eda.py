import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
raw_data = pd.read_csv("data.csv")

df = pd.DataFrame(raw_data,columns=raw_data.columns)

f = open("eda_output.txt","a")
print("Columns names of the data\n",file=f)
print(df.columns,file=f)
print("\n",file=f)

print("Info of the data\n",file=f)
df.info(buf=f,show_counts=True)
print("\n",file=f)


print("Describe of the data\n",file=f)
print(df.describe(),file=f)
print("\n",file=f)

#sample data
print(df.head(),file=f)

#types of transaction
tt = df["type"].value_counts()
plt.bar(tt.index,tt.values,color="blue",label="Transaction Type Distribution")
plt.yscale("log")
plt.savefig(r"visualization_output/transaction_types.png")
plt.show()


#Percent of fraud transaction
fraud_df = df[df["isFraud"]==1]["type"].value_counts()
non_fraud_df = df[df["isFraud"]==0]["type"].value_counts()

print(df["isFraud"].value_counts(),file=f,end="\n")


#transaction amount distribution
print(df["amount"].describe(),file=f,end="\n")

df['amount'].plot(kind='hist', bins=100, figsize=(8,4), log=True)
plt.title("Transaction Amount Distribution (Log Scale)")
plt.savefig(r"visualization_output/amt_hist.png")
plt.show()

#fraud by transaction type
prob_of_fraud = df.groupby('type')["isFraud"].mean().sort_values()
prob_of_fraud.plot(kind='bar')
print("\nProbabilties of the fraud transaction of each category",file=f)
print(prob_of_fraud,end='\n',file=f)
plt.savefig(r"visualization/fraud_types.png")
plt.show()

temp = df[df['isFraud']==1]
counts = temp.groupby('type')['isFraud'].value_counts()
print("\nNo of fraud transactions ",file=f)
print(counts,end='\n',file=f)
print(prob_of_fraud)
f.close()


df.boxplot(column='amount', by='isFraud', figsize=(8,4))
plt.yscale('symlog', linthresh=1000)
plt.title("Amount vs Fraud")
plt.savefig(r"visualization_output/fraud_vs_amt.png")
plt.show()

df[df['isFraud']==1]['amount'].plot(kind='hist',bin=100,log=True)
df[df['isFraud'] == 1]['amount'].plot(
    kind='hist',
    bins=50,
    log=True
)

plt.xlabel("Transaction Amount")
plt.ylabel("Log Frequency")
plt.savefig(r"visualization_output/Distribution of Transaction Amounts (Fraud Only)")
plt.show()

#behaiour of sender account
df['balance_change_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']

df['balance_change_orig'].hist(bins=50)
plt.savefig(r"visualization_output/Sender Balance Change Distribution.png")
plt.show()

df['balance_change_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
df['balance_change_dest'].hist(bins=50)
plt.savefig(r"visualization_output/Receiver Balance Change Distribution.png")
plt.show()



numeric_df = df[['step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFraud']]
plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.savefig(r"visualization_output/heatmp.png")
plt.show()
