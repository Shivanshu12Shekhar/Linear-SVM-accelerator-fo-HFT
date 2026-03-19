import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Fixed-point settings
Q_FBITS = 11
SCALE = 2**Q_FBITS

# ---------------------------
# 1. LOAD DATA
# ---------------------------
df = pd.read_csv("Data.csv")
print("Raw data shape:", df.shape)

# ---------------------------
# 2. LABEL SELECTION: Binary directional on LABEL_5TICK
# ---------------------------
# Strategy: predict price DIRECTION (up vs down), skip stationary rows.
# LABEL_5TICK has 65% directional rows, near-perfectly balanced (-1: 88k, +1: 90k).
# This is what a real HFT model would do — generate signal only when there's movement.
label_col = "LABEL_5TICK"
df_dir = df[df[label_col] != 0].copy()
print(f"\nUsing label: {label_col} (binary: up=+1 vs down=-1, dropping stationary 0s)")
print(f"Directional samples: {len(df_dir)} ({len(df_dir)/len(df)*100:.1f}% of data)")
print(df_dir[label_col].value_counts())

# ---------------------------
# 3. FEATURE ENGINEERING
# ---------------------------
PA = df_dir[[f"PRICE_ASK_{i}" for i in range(10)]].values
PB = df_dir[[f"PRICE_BID_{i}" for i in range(10)]].values
VA = df_dir[[f"VOLUME_ASK_{i}" for i in range(10)]].values
VB = df_dir[[f"VOLUME_BID_{i}" for i in range(10)]].values

features = {}

# --- Core microstructure ---
mid  = (PA[:, 0] + PB[:, 0]) / 2
sprd = PA[:, 0] - PB[:, 0]
features["mid_price"]   = mid
features["spread"]      = sprd
features["spread_rel"]  = sprd / (mid + 1e-8)

# --- Order book imbalance per level ---
for lvl in range(10):
    denom = VA[:, lvl] + VB[:, lvl] + 1e-8
    features[f"obi_{lvl}"] = (VB[:, lvl] - VA[:, lvl]) / denom

# --- Weighted mid-price ---
ask_wt = np.sum(PA * VA, axis=1) / (np.sum(VA, axis=1) + 1e-8)
bid_wt = np.sum(PB * VB, axis=1) / (np.sum(VB, axis=1) + 1e-8)
features["wmid"]          = (ask_wt + bid_wt) / 2
features["wmid_vs_mid"]   = features["wmid"] - mid

# --- Aggregate volume imbalance ---
tot_ask = np.sum(VA, axis=1)
tot_bid = np.sum(VB, axis=1)
features["total_ask_vol"]   = tot_ask
features["total_bid_vol"]   = tot_bid
features["total_vol_imbal"] = (tot_bid - tot_ask) / (tot_bid + tot_ask + 1e-8)

# --- Price depth slopes (ask and bid queue depth gradient) ---
levels = np.arange(10)
# Vectorized polyfit slope: sum((x-xmean)*(y-ymean)) / sum((x-xmean)^2)
x_c = levels - levels.mean()
x_var = (x_c**2).sum()
features["ask_price_slope"] = (PA - PA.mean(axis=1, keepdims=True)) @ x_c / x_var
features["bid_price_slope"] = (PB - PB.mean(axis=1, keepdims=True)) @ x_c / x_var
features["ask_vol_slope"]   = (VA - VA.mean(axis=1, keepdims=True)) @ x_c / x_var
features["bid_vol_slope"]   = (VB - VB.mean(axis=1, keepdims=True)) @ x_c / x_var

# --- Volume pressure (relative to top-of-book) ---
for lvl in range(1, 5):
    features[f"ask_pres_{lvl}"] = VA[:, lvl] / (VA[:, 0] + 1e-8)
    features[f"bid_pres_{lvl}"] = VB[:, lvl] / (VB[:, 0] + 1e-8)

# --- Relative price levels ---
for lvl in range(5):
    features[f"rel_ask_{lvl}"] = (PA[:, lvl] - mid) / (mid + 1e-8)
    features[f"rel_bid_{lvl}"] = (mid - PB[:, lvl]) / (mid + 1e-8)

# --- Log volume ratios ---
features["log_vol_ratio_0"] = np.log1p(VB[:, 0]) - np.log1p(VA[:, 0])
features["log_vol_ratio_1"] = np.log1p(VB[:, 1]) - np.log1p(VA[:, 1])
features["log_vol_ratio_2"] = np.log1p(VB[:, 2]) - np.log1p(VA[:, 2])

# --- Aggregated OBI ---
features["obi_top3"]  = sum(features[f"obi_{i}"] for i in range(3)) / 3
features["obi_top5"]  = sum(features[f"obi_{i}"] for i in range(5)) / 5
features["obi_top10"] = sum(features[f"obi_{i}"] for i in range(10)) / 10

# --- Queue imbalance acceleration (difference between near/far) ---
features["obi_near_far"] = features["obi_top3"] - (
    sum(features[f"obi_{i}"] for i in range(5, 10)) / 5
)

# Build feature matrix
feat_df = pd.DataFrame(features)
X_raw = feat_df.values
y_raw = df_dir[label_col].values

print(f"\nEngineered features: {X_raw.shape[1]}")

# ---------------------------
# 4. HANDLE NaNs & SCALE
# ---------------------------
imputer = SimpleImputer(strategy="median")
X_imp = imputer.fit_transform(X_raw)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)

# ---------------------------
# 5. TRAIN/TEST SPLIT (before any resampling to avoid leakage)
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_raw, test_size=0.2, random_state=42, stratify=y_raw
)

print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
unique, counts = np.unique(y_train, return_counts=True)
print("Train class distribution:", dict(zip(unique.tolist(), counts.tolist())))

# ---------------------------
# 6. SMOTE ON TRAINING SET (manual, no external lib)
# ---------------------------
def smote_balance(X, y, random_state=42):
    """Oversample minority class to match majority using SMOTE interpolation."""
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(random_state)
    unique, counts = np.unique(y, return_counts=True)
    majority_count = counts.max()

    print("\nSMOTE balancing...")
    print("Before:", dict(zip(unique.tolist(), counts.tolist())))

    X_parts, y_parts = [X.copy()], [y.copy()]
    for cls, cnt in zip(unique, counts):
        if cnt == majority_count:
            continue
        n_needed = majority_count - cnt
        X_cls = X[y == cls]
        k = min(5, len(X_cls) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X_cls)
        _, nbr_indices = nn.kneighbors(X_cls)
        nbr_indices = nbr_indices[:, 1:]

        i_rand    = rng.integers(0, len(X_cls), size=n_needed)
        nn_choice = np.array([rng.choice(nbr_indices[i]) for i in i_rand])
        alpha     = rng.uniform(0, 1, size=(n_needed, 1))
        synthetic = X_cls[i_rand] + alpha * (X_cls[nn_choice] - X_cls[i_rand])

        X_parts.append(synthetic)
        y_parts.append(np.full(n_needed, cls))

    X_out = np.vstack(X_parts)
    y_out = np.concatenate(y_parts)
    unique2, counts2 = np.unique(y_out, return_counts=True)
    print("After:", dict(zip(unique2.tolist(), counts2.tolist())))
    return X_out, y_out

X_train_bal, y_train_bal = smote_balance(X_train, y_train)

# Shuffle
perm = np.random.default_rng(0).permutation(len(X_train_bal))
X_train_bal = X_train_bal[perm]
y_train_bal = y_train_bal[perm]

# ---------------------------
# 7. PCA -> 16 COMPONENTS
# ---------------------------
pca = PCA(n_components=16, random_state=42)
X_train_pca = pca.fit_transform(X_train_bal)
X_test_pca  = pca.transform(X_test)

print(f"\nPCA variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")
print("Number of PCA features chosen:", 16)
N_FEATURES = 16

# ---------------------------
# 8. TRAIN SVM
# ---------------------------
clf = LinearSVC(loss="squared_hinge", max_iter=5000, random_state=42)
clf.fit(X_train_pca, y_train_bal)

W = clf.coef_[0]   # binary: single weight vector
b = clf.intercept_[0]

# ---------------------------
# 9. EVALUATION
# ---------------------------
y_pred_float = clf.predict(X_test_pca)
acc_float = accuracy_score(y_test, y_pred_float)
print(f"\nFloat Accuracy: {acc_float:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_float))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_float))

# ---------------------------
# 10. FIXED-POINT CONVERSION & EXPORT
# ---------------------------
def quantize(val):
    return np.round(val * SCALE).astype(np.int16)

W_fixed = quantize(W)
b_fixed = int(round(b * SCALE * SCALE))

with open("weights.mem", "w") as f:
    for w in W_fixed:
        val = int(w) & 0xFFFF
        f.write(f"{val:04X}\n")

with open("bias.mem", "w") as f:
    val = int(b_fixed) & 0xFFFFFFFF
    f.write(f"{val:08X}\n")

# ---------------------------
# 11. FIXED-POINT SIMULATION
# ---------------------------
def hex_to_int16(h):
    v = int(h, 16)
    return v - 65536 if v >= 32768 else v

def hex_to_int32(h):
    v = int(h, 16)
    return v - 2**32 if v >= 2**31 else v

W_fixed_loaded = np.array([hex_to_int16(x) for x in open("weights.mem").read().splitlines()], dtype=np.int16)
b_fixed_loaded = np.int64(hex_to_int32(open("bias.mem").read().strip()))

X_fixed_all = np.round(X_test_pca * SCALE).astype(np.int16)
pred_fixed = []
for i in range(len(X_fixed_all)):
    x = X_fixed_all[i]
    acc_hw = np.int64(0)
    for j in range(N_FEATURES):
        acc_hw += np.int64(W_fixed_loaded[j]) * np.int64(x[j])
    acc_hw += b_fixed_loaded
    pred_fixed.append(1 if acc_hw >= 0 else -1)

acc_fixed = accuracy_score(y_test, pred_fixed)
print(f"\nFixed Accuracy (hardware simulation): {acc_fixed:.4f}")
print(f"Accuracy Drop: {(acc_float - acc_fixed)*100:.4f}%")

# ---------------------------
# 12. EXPORT 100 STRATIFIED TEST VECTORS
# ---------------------------
num_test_vectors = 100
pred_fixed = np.array(pred_fixed)
y_pred_float_arr = np.array(y_pred_float)

# Stratified sample: 50 up, 50 down from TRUE labels
rng = np.random.default_rng(42)
idx_up   = np.where(y_test == 1)[0]
idx_down = np.where(y_test == -1)[0]
sel_up   = rng.choice(idx_up,   size=50, replace=False)
sel_down = rng.choice(idx_down, size=50, replace=False)
selected = np.concatenate([sel_up, sel_down])
rng.shuffle(selected)

print(f"\nExporting {num_test_vectors} stratified test vectors (50 up / 50 down by true label)...")
print("True distribution:", dict(zip(*np.unique(y_test[selected], return_counts=True))))
print("HW pred distribution:", dict(zip(*np.unique(pred_fixed[selected], return_counts=True))))

with open("input_vectors.mem", "w") as f_in, \
     open("expected_outputs.mem", "w") as f_out, \
     open("true_labels.mem", "w") as f_true:

    for i in selected:
        x = X_fixed_all[i]
        line_hex = ""
        for feat_idx in range(N_FEATURES - 1, -1, -1):
            val = int(x[feat_idx]) & 0xFFFF
            line_hex += f"{val:04X}"
        f_in.write(f"{line_hex}\n")

        # expected_outputs: HW predicted label  (1=up, 0=down)
        f_out.write(f"{'1' if pred_fixed[i] == 1 else '0'}\n")

        # true_labels: ground truth             (1=up, 0=down)
        f_true.write(f"{'1' if y_test[i] == 1 else '0'}\n")

# CSV summary with all three values
import csv
with open("test_vectors_summary.csv", "w", newline="") as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow(["vector_idx", "true_label", "hw_pred", "float_pred",
                     "hw_correct", "float_correct"])
    for out_i, i in enumerate(selected):
        true_lbl  = int(y_test[i])
        hw_lbl    = int(pred_fixed[i])
        fl_lbl    = int(y_pred_float_arr[i])
        writer.writerow([out_i, true_lbl, hw_lbl, fl_lbl,
                         int(hw_lbl == true_lbl), int(fl_lbl == true_lbl)])

print("\nFiles exported:")
print("  weights.mem             - 16 fixed-point weights (Q11)")
print("  bias.mem                - Fixed-point bias (Q22)")
print("  input_vectors.mem       - 100 test vectors (hex, feat[15] first)")
print("  expected_outputs.mem    - HW predicted label (1=up, 0=down)")
print("  true_labels.mem         - Ground truth label  (1=up, 0=down)")
print("  test_vectors_summary.csv - Full comparison table")
print(f"\nExported {num_test_vectors} test vectors.")
print("Training done using real dataset")
