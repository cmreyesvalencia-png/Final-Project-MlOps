import pandas as pd

df = pd.read_csv('C:/C22/FP/creditcard.csv')

print("="*50)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("="*50)
print(f"Total transactions: {len(df):,}")
print(f"Fraud cases: {df['Class'].sum():,} ({df['Class'].mean()*100:.4f}%)")
print(f"Normal cases: {(df['Class']==0).sum():,}")
print(f"Fraud percentage: {df['Class'].mean()*100:.4f}%")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Number of features: {df.shape[1]}")
print(f"Features: V1-V28 (28 features), Time, Amount, Class")
print("")
print("Class Distribution:")
print(f"  Class 0 (Normal):  {(df['Class']==0).sum():,} ({((df['Class']==0).sum()/len(df))*100:.2f}%)")
print(f"  Class 1 (Fraud):   {df['Class'].sum():,} {((df['Class'].sum()/len(df))*100):.4f}%)")
print("="*50)
