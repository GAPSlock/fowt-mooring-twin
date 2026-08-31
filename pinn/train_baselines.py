import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

TRAIN_DATA = r'C:\Users\Guramrit Pal Singh\OneDrive\Desktop\Research\fowt-mooring-twin\data\training\train_data.h5'
TEST_DATA = r'C:\Users\Guramrit Pal Singh\OneDrive\Desktop\Research\fowt-mooring-twin\data\training\test_data.h5'

print("Loading data...")
df_train = pd.read_hdf(TRAIN_DATA)
df_test = pd.read_hdf(TEST_DATA)

def get_features_targets(df):
    X = df[['PtfmSurge_[m]', 'PtfmSway_[m]', 'PtfmHeave_[m]', 
            'PtfmRoll_[deg]', 'PtfmPitch_[deg]', 'PtfmYaw_[deg]']].values
    Y = df[['FAIRTEN1_[N]', 'FAIRTEN2_[N]', 'FAIRTEN3_[N]']].values / 1000.0
    dt = 0.0125
    V = np.gradient(X, dt, axis=0)
    A = np.gradient(V, dt, axis=0)
    X_full = np.concatenate([X, V, A], axis=1)
    return X_full, Y

X_train, Y_train = get_features_targets(df_train)
X_test, Y_test = get_features_targets(df_test)

# Focus on Line 2 (Surge-aligned) for baselines, matching the paper table
y_train = Y_train[:, 1]
y_test = Y_test[:, 1]

print("\n--- Training Linear Regression ---")
lr = LinearRegression()
lr.fit(X_train, y_train)

t0 = time.time()
lr_preds = lr.predict(X_test)
t1 = time.time()
lr_time = (t1 - t0) / len(X_test) * 1000 # ms per sample
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
lr_r2 = r2_score(y_test, lr_preds)

print(f"LR RMSE: {lr_rmse:.1f} kN")
print(f"LR R2:   {lr_r2:.3f}")
print(f"LR Time: {lr_time:.3f} ms")

print("\n--- Training Random Forest ---")
# Limit max samples to 100k so it trains fast, max_depth 20, 100 trees
rf = RandomForestRegressor(n_estimators=100, max_depth=20, n_jobs=-1, max_samples=100000)
rf.fit(X_train, y_train)

t0 = time.time()
rf_preds = rf.predict(X_test)
t1 = time.time()
rf_time = (t1 - t0) / len(X_test) * 1000 # ms per sample
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_r2 = r2_score(y_test, rf_preds)

print(f"RF RMSE: {rf_rmse:.1f} kN")
print(f"RF R2:   {rf_r2:.3f}")
print(f"RF Time: {rf_time:.3f} ms")

with open('baseline_results.txt', 'w') as f:
    f.write(f"LR: RMSE={lr_rmse:.1f}, R2={lr_r2:.3f}, Time={lr_time:.3f}ms\n")
    f.write(f"RF: RMSE={rf_rmse:.1f}, R2={rf_r2:.3f}, Time={rf_time:.3f}ms\n")
