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
def make_ratio_feature(X, num_idx, den_idx, eps=1e-8):
    """构建比例特征 (分子列 / 分母列)"""
    return X[:, num_idx] / (X[:, den_idx] + eps)

# Step 5 - append_column
def append_column(X, col):
    """将新特征向量作为最后一列附加到特征矩阵中"""
    col_arr = np.asarray(col).reshape(-1, 1)
    return np.hstack([X, col_arr])

# Step 6 - one_hot_encode (not yet solved)
# TODO: implement

# Step 7 - fit_standardizer (not yet solved)
# TODO: implement

# Step 8 - apply_standardizer (not yet solved)
# TODO: implement

# Step 9 - add_bias_column (not yet solved)
# TODO: implement

# Step 10 - make_shuffled_indices (not yet solved)
# TODO: implement

# Step 11 - partition_indices (not yet solved)
# TODO: implement

# Step 12 - subset_xy (not yet solved)
# TODO: implement

# Step 13 - ols_fit (not yet solved)
# TODO: implement

# Step 14 - ols_predict (not yet solved)
# TODO: implement

# Step 15 - mean_absolute_error (not yet solved)
# TODO: implement

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

