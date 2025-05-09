import numpy as np
import pandas as pd
import torch
import os
import random
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from model.wgf_imp import NeuralGradFlowImputer

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seed
set_seed(42)

# 1. Load the data
print("Loading data...")
data = pd.read_csv("20250402_total_data.csv")
print(f"Data shape: {data.shape}")

# 2. Normalize the data using MaxAbsScaler
print("Normalizing data...")
scaler = MaxAbsScaler()
normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
print("Data normalized")

# Save the normalized data for reference
normalized_data.to_csv("normalized_data.csv", index=False)
print("Normalized data saved to normalized_data.csv")

# 3. Randomly introduce 30% missing values in each column
print("Introducing missing values...")
missing_mask = np.zeros(normalized_data.shape, dtype=bool)
for col in range(normalized_data.shape[1]):
    # Generate random indices for missing values (30% of the data)
    missing_indices = np.random.choice(
        normalized_data.shape[0],
        size=int(normalized_data.shape[0] * 0.3),
        replace=False
    )
    missing_mask[missing_indices, col] = True

# Create a copy of the normalized data with missing values
data_with_missing = normalized_data.copy()
data_with_missing[missing_mask] = np.nan

# Save the data with missing values
data_with_missing.to_csv("data_with_missing.csv", index=False)
print("Data with missing values saved to data_with_missing.csv")

# 4. Use NeuralGradFlowImputer to impute the missing data
print("Imputing missing data...")
# Convert to numpy arrays for the imputer
X_miss = data_with_missing.values
X_true = normalized_data.values

# 确保数据是浮点型
X_miss = torch.tensor(X_miss, dtype=torch.float32)
X_true = torch.tensor(X_true, dtype=torch.float32)

# 验证数据
print("验证数据格式...")
print(f"X_miss 形状: {X_miss.shape}")
print(f"X_miss 数据类型: {X_miss.dtype}")
print(f"X_miss 是否包含 NaN: {torch.isnan(X_miss).any()}")
print(f"X_true 形状: {X_true.shape}")
print(f"X_true 数据类型: {X_true.dtype}")
print(f"X_true 是否包含 NaN: {torch.isnan(X_true).any()}")

# Set up the NeuralGradFlowImputer
model = NeuralGradFlowImputer(
    entropy_reg=10.0,
    bandwidth=10.0,
    score_net_epoch=2000,
    niter=50,
    initializer=None,
    mlp_hidden=[256, 256],
    lr=1.0e-1,
    score_net_lr=1.0e-3,
    device=device
)

# Impute the missing values
print("Starting imputation process...")
X_imputed = model.fit_transform(X_miss, X_true=X_true, verbose=True, report_interval=5)

# Convert the imputed data to a DataFrame
imputed_data = pd.DataFrame(X_imputed.detach().cpu().numpy(), columns=data.columns)

# Save the imputed data
imputed_data.to_csv("imputed_data.csv", index=False)
print("Imputed data saved to imputed_data.csv")

# 5. Compare the imputed data with the original normalized data
print("Calculating metrics...")

# Calculate MAE
mae = mean_absolute_error(
    normalized_data.values[missing_mask],
    imputed_data.values[missing_mask]
)
print(f"MAE: {mae:.4f}")

# Calculate RMSE
rmse = math.sqrt(mean_squared_error(
    normalized_data.values[missing_mask],
    imputed_data.values[missing_mask]
))
print(f"RMSE: {rmse:.4f}")

# Save the results
results = {
    "MAE": [mae],
    "RMSE": [rmse]
}
pd.DataFrame(results).to_csv("imputation_results.csv", index=False)
print("Results saved to imputation_results.csv")

print("Task completed successfully!")
