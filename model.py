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
import numpy as np

def make_ratio_feature(numerator, denominator, eps=1e-8) -> np.ndarray:
    """
    通过安全除法衍生比例特征
    :param numerator: 分子一维数组，形状 (N,)
    :param denominator: 分母一维数组，形状 (N,)
    :param eps: 防止除以零的微小浮点数偏置 (默认 1e-8)
    :return: 比例特征一维数组，形状 (N,)
    """
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    return numerator / (denominator + eps)

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

# Step 16 - root_mean_squared_error
# ── Step 016  root_mean_squared_error ──
def root_mean_squared_error(y_true, y_pred):
    """均方根误差 RMSE"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

# Step 17 - r_squared
import numpy as np

def r_squared(y_true, y_pred) -> float:
    """
    计算决定系数 R^2
    :param y_true: 真实目标值一维数组
    :param y_pred: 模型预测值一维数组
    :return: R^2 分数 (若 SS_tot 为 0 则返回 0.0)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    # 残差平方和
    ss_res = np.sum((y_true - y_pred) ** 2)
    # 总平方和
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # 边界情况处理：真实值全相等导致方差为 0
    if ss_tot == 0.0:
        return 0.0
        
    return float(1.0 - (ss_res / ss_tot))

# Step 18 - residual_summary
import numpy as np

def residual_summary(y_true, y_pred) -> dict:
    """
    计算残差指标摘要字典
    严格包含键: 'mean', 'std', 'median_abs'
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # 残差 r = y_true - y_pred
    r = y_true - y_pred
    
    return {
        'mean': float(np.mean(r)),
        'std': float(np.std(r)),                   # 总体标准差 (ddof=0)
        'median_abs': float(np.median(np.abs(r)))  # 绝对残差中位数
    }

# Step 19 - prepare_cleaned_features
import numpy as np

def prepare_cleaned_features(X, iqr_k=1.5):
    """
    依次调用已有的三项辅助函数完成缺失值填充与异常值截断
    """
    # 1. 均值插补
    X_imputed = impute_nan_with_mean(X)
    # 2. 计算 IQR 边界
    lower, upper = compute_iqr_bounds(X_imputed, iqr_k)
    # 3. 截断异常值
    return clip_columns(X_imputed, lower, upper)

# Step 20 - assemble_feature_matrix
import numpy as np

def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    """
    组装扩展特征矩阵：
    1. 提取分子列与分母列，通过 make_ratio_feature 构建比例特征
    2. 使用 append_column 将比例特征拼接到右侧
    3. 若 cat_labels 不为 None，通过 one_hot_encode 编码并水平拼接到右侧
    """
    # 提取对应列并计算比例特征
    num = X_num[:, ratio_num_idx]
    den = X_num[:, ratio_den_idx]
    ratio = make_ratio_feature(num, den)
    
    # 拼接入数值矩阵
    X_ext = append_column(X_num, ratio)
    
    # 拼接类别独热特征
    if cat_labels is not None:
        cat_one_hot = one_hot_encode(cat_labels)
        X_ext = np.hstack([X_ext, cat_one_hot])
        
    return X_ext

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    """
    打乱并划分 Train、Validation、Test 数据集
    返回包含 6 个键的字典: 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'
    """
    indices = make_shuffled_indices(len(X), seed)
    train_idx, val_idx, test_idx = partition_indices(indices, train_ratio, val_ratio)
    
    X_train, y_train = subset_xy(X, y, train_idx)
    X_val, y_val = subset_xy(X, y, val_idx)
    X_test, y_test = subset_xy(X, y, test_idx)
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }

# Step 22 - standardize_and_add_bias
def standardize_and_add_bias(splits):
    """
    仅使用 X_train 计算标准化参数，应用到所有分割集并前置拼接偏置列
    返回: (std_splits, mean, std)
    """
    mean, std = fit_standardizer(splits['X_train'])
    
    X_train_std = add_bias_column(apply_standardizer(splits['X_train'], mean, std))
    X_val_std = add_bias_column(apply_standardizer(splits['X_val'], mean, std))
    X_test_std = add_bias_column(apply_standardizer(splits['X_test'], mean, std))
    
    std_splits = {
        'X_train': X_train_std,
        'y_train': splits['y_train'],
        'X_val': X_val_std,
        'y_val': splits['y_val'],
        'X_test': X_test_std,
        'y_test': splits['y_test']
    }
    
    return std_splits, mean, std

# Step 23 - evaluate_predictions
def evaluate_predictions(y_true, y_pred):
    """
    返回包含四个指定键的度量字典: 'mae', 'rmse', 'r2', 'residual_summary'
    """
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'r2': r_squared(y_true, y_pred),
        'residual_summary': residual_summary(y_true, y_pred)
    }

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

