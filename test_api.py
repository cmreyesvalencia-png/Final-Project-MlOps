import pandas as pd
import requests

print("="*50)
print("API TEST")
print("="*50)

df = pd.read_csv('C:/C22/FP/creditcard.csv')

# Test normal transaction
normal = df[df['Class']==0].iloc[0, :-1].tolist()
resp = requests.post('http://localhost:8000/predict', json={'features': normal})
print(f'Normal Transaction: {resp.json()}')

# Test fraud transaction
fraud = df[df['Class']==1].iloc[0, :-1].tolist()
resp = requests.post('http://localhost:8000/predict', json={'features': fraud})
print(f'Fraud Transaction: {resp.json()}')

print("="*50)
print("SUCCESS - API is working!")
print("="*50)
