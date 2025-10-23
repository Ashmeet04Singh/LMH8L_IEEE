import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.combine import SMOTETomek
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
RANDOM_STATE = 42
N_FOLDS = 8  # CPU-optimized for ~6 hours
WORK_DIR = r"C:\Users\Admin\OneDrive\Desktop\IEEE"
os.chdir(WORK_DIR)

# ---------- LOAD DATA ----------
train = pd.read_csv("main.csv")
test = pd.read_csv("test.csv")
id_col = "sha256"
target_col = "Label"

train_ids = train[id_col].copy()
test_ids = test[id_col].copy()
X = train.drop(columns=[id_col,target_col])
y = train[target_col].astype(int)
X_test = test.drop(columns=[id_col])

# Align columns
common_cols = [c for c in X.columns if c in X_test.columns]
X = X[common_cols].copy()
X_test = X_test[common_cols].copy()

# Numeric only
X = X.select_dtypes(include=[np.number])
X_test = X_test[X.columns]

# ---------- MISSING & INF ----------
imputer = SimpleImputer(strategy="median")
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X.columns)

# Missing features
for df in [X_imp, X_test_imp]:
    df["missing_count"] = df.isnull().sum(axis=1)
    df["has_missing"] = (df["missing_count"]>0).astype(int)

X_imp.replace([np.inf,-np.inf],0,inplace=True)
X_test_imp.replace([np.inf,-np.inf],0,inplace=True)
X_imp.fillna(0,inplace=True)
X_test_imp.fillna(0,inplace=True)

# ---------- FEATURE ENGINEERING ----------
def add_row_features(df):
    df = df.copy()
    df["r_sum"] = df.sum(axis=1)
    df["r_mean"] = df.mean(axis=1)
    df["r_std"] = df.std(axis=1)
    df["r_max"] = df.max(axis=1)
    df["r_min"] = df.min(axis=1)
    df["r_range"] = df["r_max"] - df["r_min"]
    df["r_zero_count"] = (df==0).sum(axis=1)
    return df

X_feat = add_row_features(X_imp)
X_test_feat = add_row_features(X_test_imp)

# Drop constant columns
const_cols = X_feat.columns[X_feat.nunique()<=1]
X_feat.drop(columns=const_cols,inplace=True)
X_test_feat.drop(columns=const_cols,inplace=True)

# ---------- SCALE ----------
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_feat)
X_test_scaled = scaler.transform(X_test_feat)

# ---------- RESAMPLING ----------
sm = SMOTETomek(random_state=RANDOM_STATE)
X_res, y_res = sm.fit_resample(X_scaled, y)

# ---------- BASE MODEL PARAMETERS ----------
xgb_params = dict(n_estimators=500,max_depth=10,learning_rate=0.02,
                  subsample=0.9,colsample_bytree=0.8,random_state=RANDOM_STATE,
                  use_label_encoder=False,eval_metric='logloss',n_jobs=-1,verbosity=0)

lgb_params = dict(n_estimators=500,max_depth=10,learning_rate=0.02,
                  subsample=0.9,colsample_bytree=0.8,random_state=RANDOM_STATE,n_jobs=-1)

cat_params = dict(iterations=1200,depth=10,learning_rate=0.025,
                  l2_leaf_reg=3,random_seed=RANDOM_STATE,verbose=0,bootstrap_type='Bernoulli')

rf_params = dict(n_estimators=400,max_depth=12,random_state=RANDOM_STATE,n_jobs=-1)

mlp_params = dict(hidden_layer_sizes=(256,128),max_iter=800,random_state=RANDOM_STATE,alpha=0.0005)

# ---------- K-FOLD OOF ----------
skf = StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=RANDOM_STATE)
n_models = 5
oof = np.zeros((X_res.shape[0], n_models))
test_preds = np.zeros((X_test_scaled.shape[0], n_models))

for fold,(tr_idx,val_idx) in enumerate(skf.split(X_res,y_res),start=1):
    print(f"\n=== Fold {fold}/{N_FOLDS} ===")
    X_tr, X_val = X_res[tr_idx], X_res[val_idx]
    y_tr, y_val = y_res[tr_idx], y_res[val_idx]

    # XGB
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_tr,y_tr)
    oof[val_idx,0] = xgb.predict_proba(X_val)[:,1]
    test_preds[:,0] += xgb.predict_proba(X_test_scaled)[:,1]/N_FOLDS

    # LGB
    lgb = LGBMClassifier(**lgb_params)
    lgb.fit(X_tr,y_tr)
    oof[val_idx,1] = lgb.predict_proba(X_val)[:,1]
    test_preds[:,1] += lgb.predict_proba(X_test_scaled)[:,1]/N_FOLDS

    # CatBoost
    cat = CatBoostClassifier(**cat_params)
    cat.fit(X_tr,y_tr,eval_set=(X_val,y_val),verbose=False)
    oof[val_idx,2] = cat.predict_proba(X_val)[:,1]
    test_preds[:,2] += cat.predict_proba(X_test_scaled)[:,1]/N_FOLDS

    # RandomForest
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_tr,y_tr)
    oof[val_idx,3] = rf.predict_proba(X_val)[:,1]
    test_preds[:,3] += rf.predict_proba(X_test_scaled)[:,1]/N_FOLDS

    # MLP
    mlp = MLPClassifier(**mlp_params)
    mlp.fit(X_tr,y_tr)
    oof[val_idx,4] = mlp.predict_proba(X_val)[:,1]
    test_preds[:,4] += mlp.predict_proba(X_test_scaled)[:,1]/N_FOLDS

    # Fold F1 check
    blend_fold = np.dot(oof[val_idx],[0.25,0.25,0.25,0.15,0.10])
    f_fold = f1_score(y_val,(blend_fold>0.5).astype(int),average='weighted')
    print(f"Fold blended F1: {f_fold:.6f}")

# ---------- AUTOMATED WEIGHT OPTIMIZATION ----------
def f1_loss(weights):
    blend = np.dot(oof,weights)
    return -f1_score(y_res,(blend>0.5).astype(int),average='weighted')

res = minimize(f1_loss,[0.2]*n_models,bounds=[(0,1)]*n_models,
               constraints={'type':'eq','fun':lambda w: w.sum()-1})
best_weights = res.x
print("Best ensemble weights:",best_weights)

blend_oof = np.dot(oof,best_weights)
blend_test = np.dot(test_preds,best_weights)

# ---------- LEVEL 2 META-LEARNER ----------
top_features_idx = np.argsort(X_feat.var())[-10:]  # top 10 variance features
X_meta = np.hstack([blend_oof.reshape(-1,1), X_res[:,top_features_idx]])
X_test_meta = np.hstack([blend_test.reshape(-1,1), X_test_scaled[:,top_features_idx]])

meta = CatBoostClassifier(iterations=1200,depth=8,learning_rate=0.025,
                          random_seed=RANDOM_STATE,verbose=0)
meta.fit(X_meta,y_res)
oof_meta = meta.predict_proba(X_meta)[:,1]
test_meta = meta.predict_proba(X_test_meta)[:,1]

# ---------- THRESHOLD SEARCH ----------
best_t,best_f1 = 0.5,0
for t in np.linspace(0.3,0.7,2001):  # fine-grained
    f = f1_score(y_res,(oof_meta>t).astype(int),average='weighted')
    if f>best_f1:
        best_f1=f
        best_t=t
print(f"Optimized threshold: {best_t:.4f}, OOF F1: {best_f1:.6f}")

# ---------- OOF DIAGNOSTICS ----------
oof_labels = (oof_meta>best_t).astype(int)
print("\nClassification report (OOF):")
print(classification_report(y_res,oof_labels,digits=6))

# ---------- TEST PREDICTIONS ----------
final_test_preds = (test_meta>best_t).astype(int)
submission = pd.DataFrame({id_col:test_ids,"Label":final_test_preds})
submission.to_csv("LMH8L.csv",index=False) 
print("\nSaved LMH8L.csv")
print(submission.head(10))
