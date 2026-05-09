import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("TRAINING WITH HYPERPARAMETER TUNING")
print("="*60)

# Load data
print("\n[1/5] Loading data...")
df = pd.read_csv('C:/C22/FP/creditcard.csv')
print(f"   Data: {len(df):,} transactions")
print(f"   Fraud: {df['Class'].sum():,} ({df['Class'].mean()*100:.4f}%)")

# Prepare data
print("\n[2/5] Preprocessing...")
X = df.drop('Class', axis=1)
y = df['Class']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("   Features normalized")

# Split data
print("\n[3/5] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"   Training: {len(X_train):,}")
print(f"   Testing: {len(X_test):,}")

# Handle imbalance
scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
print(f"\n   Scale_pos_weight: {scale_pos_weight:.2f}")

# ============================================
# HYPERPARAMETER TUNING WITH RANDOMIZED SEARCH
# ============================================
print("\n[4/5] Hyperparameter Tuning with RandomizedSearchCV...")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='auc'
)

random_search = RandomizedSearchCV(
    xgb_model,
    param_grid,
    n_iter=10,  # Try 10 random combinations
    cv=3,       # 3-fold cross validation
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("   Searching for best parameters...")
random_search.fit(X_train, y_train)

print("\n   ✓ Best parameters found:")
for param, value in random_search.best_params_.items():
    print(f"      {param}: {value}")

# ============================================
# TRAIN FINAL MODEL WITH BEST PARAMETERS
# ============================================
print("\n[5/5] Training final model with best parameters...")

best_model = xgb.XGBClassifier(
    **random_search.best_params_,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='auc'
)

best_model.fit(X_train, y_train, verbose=False)
print("   Model training complete!")

# ============================================
# EVALUATION
# ============================================
print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)

y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nPrecision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"   True Negatives: {cm[0,0]:,}")
print(f"   False Positives: {cm[0,1]:,}")
print(f"   False Negatives: {cm[1,0]:,}")
print(f"   True Positives: {cm[1,1]:,}")

# Save model and scaler
with open('fraud_model_tuned.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('scaler_tuned.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n✅ Model saved as: fraud_model_tuned.pkl")
print("✅ Scaler saved as: scaler_tuned.pkl")
print("="*60)
