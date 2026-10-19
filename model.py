"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    """用每列的非空均值填补 NaN"""
    X_imputed = np.array(X, dtype=float, copy=True)
    col_means = np.nanmean(X_imputed, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask = np.isnan(X_imputed)
    col_indices = np.where(nan_mask)[1]
    X_imputed[nan_mask] = col_means[col_indices]
    return X_imputed

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    """基于四分位距 (IQR) 计算每列截断边界"""
    q25 = np.percentile(X, 25, axis=0)
    q75 = np.percentile(X, 75, axis=0)
    iqr = q75 - q25
    lower = q25 - k * iqr
    upper = q75 + k * iqr
    return lower, upper

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    """将特征矩阵各列的值限制在 [lower, upper] 范围内"""
    return np.clip(X, lower, upper)

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    return X[:, denominator] / (X[:, denominator] + eps)

# Step 5 - append_column
def append_column(X, col):
    """将新特征向量作为最后一列附加到特征矩阵中"""
    col_arr = np.asarray(col).reshape(-1, 1)
    return np.hstack([X, col_arr])

# Step 6 - one_hot_encode
def one_hot_encode(cat_labels, categories=None):
    """类别特征的独热编码 (One-Hot Encoding)"""
    cat_arr = np.asarray(cat_labels)
    if categories is None:
        categories = np.unique(cat_arr)
    return (cat_arr[:, None] == categories).astype(float)

# Step 7 - fit_standardizer
def fit_standardizer(X):
    """在训练集上统计各特征列的均值与标准差"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return mean, std

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    """应用 Z-score 标准化变换"""
    return (X - mean) / std

# Step 9 - add_bias_column
def add_bias_column(X):
    """在特征矩阵最前列拼接常数项 1 作为偏置 (截距项)"""
    return np.column_stack([np.ones(X.shape[0]), X])

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n, seed=None):
    """生成可复现的随机重排索引"""
    if seed is not None:
        np.random.seed(seed)
    return np.random.permutation(n)

# Step 11 - partition_indices
def partition_indices(indices, train_ratio=0.7, val_ratio=0.15):
    """将索引按比例切分为训练集、验证集和测试集"""
    n = len(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return train_idx, val_idx, test_idx

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    """根据索引切片特征矩阵 X 与目标向量 y"""
    return X[indices], y[indices]

# Step 13 - ols_fit
def ols_fit(X, y):
    """利用最小二乘法正规方程/伪逆求解模型权重"""
    return np.linalg.pinv(X) @ y

# Step 14 - ols_predict
def ols_predict(X, w):
    """模型预测值计算: y_hat = X @ w"""
    return X @ w

# Step 15 - mean_absolute_error
# ── Step 015  mean_absolute_error ──
def mean_absolute_error(y_true, y_pred):
    """平均绝对误差 MAE"""
    return float(np.mean(np.abs(y_true - y_pred)))

# Step 16 - root_mean_squared_error (not yet solved)
# TODO: implement

# Step 17 - r_squared (not yet solved)
# TODO: implement

# Step 18 - residual_summary (not yet solved)
# TODO: implement

# Step 19 - prepare_cleaned_features (not yet solved)
# TODO: implement

# Step 20 - assemble_feature_matrix (not yet solved)
# TODO: implement

# Step 21 - make_train_val_test (not yet solved)
# TODO: implement

# Step 22 - standardize_and_add_bias (not yet solved)
# TODO: implement

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

