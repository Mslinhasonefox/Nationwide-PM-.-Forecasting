# 1. 导入必要库
# ------------------------
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# ------------------------
# 2. 读取数据集
# ------------------------
train_df = pd.read_csv('/content/drive/MyDrive/Station_split/gtnnwr_train_beta.csv')
val_df = pd.read_csv('/content/drive/MyDrive/Station_split/gtnnwr_val_beta.csv')
test_df = pd.read_csv('/content/drive/MyDrive/Station_split/gtnnwr_test_beta.csv')

# 【合并】train + val
train_all_df = pd.concat([train_df, val_df], ignore_index=True)

# ------------------------
# 3. 特征列定义
# ------------------------
var_cols = ["AOD", "Surface_Pressure", "Temperature", "Wind_Speed",
            "Wind_Direction", "Relative_Humidity", "Planetary_Boundary_Height", "NDVI", "DEM"]
beta_cols = [f'beta_{c}' for c in var_cols]
target_col = "PM25"

# ------------------------
# 4. 伪图构建函数
# ------------------------
def build_pseudo_image(df, var_cols, beta_cols, time_steps=7):
    df = df.sort_values(["year", "month", "day", "stationnum"]).reset_index(drop=True)
    X_vars, X_betas, X_coords, y = [], [], [], []
    for station in df['stationnum'].unique():
        station_df = df[df['stationnum'] == station]
        if len(station_df) < time_steps + 1:
            continue
        for i in range(len(station_df) - time_steps):
            vars_seq = station_df.iloc[i:i+time_steps][var_cols].values
            betas_seq = station_df.iloc[i:i+time_steps][beta_cols].values
            coords_seq = station_df.iloc[i:i+time_steps][['longitude', 'latitude']].values
            target = station_df.iloc[i+time_steps][target_col]
            X_vars.append(vars_seq)
            X_betas.append(betas_seq)
            X_coords.append(coords_seq)
            y.append(target)
    return np.array(X_vars), np.array(X_betas), np.array(X_coords), np.array(y).reshape(-1, 1)

# ------------------------
# 5. 标准化处理
# ------------------------
time_steps = 7
X_train_vars, X_train_betas, X_train_coords, y_train = build_pseudo_image(train_all_df, var_cols, beta_cols, time_steps)
X_test_vars,  X_test_betas,  X_test_coords,  y_test  = build_pseudo_image(test_df, var_cols, beta_cols, time_steps)

# 先扁平化
scaler_vars = StandardScaler()
scaler_betas = StandardScaler()
scaler_coords = StandardScaler()

X_train_vars_flat = X_train_vars.reshape(-1, X_train_vars.shape[-1])
X_train_betas_flat = X_train_betas.reshape(-1, X_train_betas.shape[-1])
X_train_coords_flat = X_train_coords.reshape(-1, X_train_coords.shape[-1])

X_train_vars_scaled = scaler_vars.fit_transform(X_train_vars_flat).reshape(X_train_vars.shape)
X_train_betas_scaled = scaler_betas.fit_transform(X_train_betas_flat).reshape(X_train_betas.shape)
X_train_coords_scaled = scaler_coords.fit_transform(X_train_coords_flat).reshape(X_train_coords.shape)

X_test_vars_flat = X_test_vars.reshape(-1, X_test_vars.shape[-1])
X_test_betas_flat = X_test_betas.reshape(-1, X_test_betas.shape[-1])
X_test_coords_flat = X_test_coords.reshape(-1, X_test_coords.shape[-1])

X_test_vars_scaled = scaler_vars.transform(X_test_vars_flat).reshape(X_test_vars.shape)
X_test_betas_scaled = scaler_betas.transform(X_test_betas_flat).reshape(X_test_betas.shape)
X_test_coords_scaled = scaler_coords.transform(X_test_coords_flat).reshape(X_test_coords.shape)

# ------------------------
# 6. 构建Dataset
# ------------------------
class PseudoDataset(Dataset):
    def __init__(self, X_vars, X_betas, X_coords, y):
        self.X_vars = torch.tensor(X_vars, dtype=torch.float32)
        self.X_betas = torch.tensor(X_betas, dtype=torch.float32)
        self.X_coords = torch.tensor(X_coords, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_vars[idx], self.X_betas[idx], self.X_coords[idx], self.y[idx]

train_dataset = PseudoDataset(X_train_vars_scaled, X_train_betas_scaled, X_train_coords_scaled, y_train)
test_dataset  = PseudoDataset(X_test_vars_scaled,  X_test_betas_scaled,  X_test_coords_scaled,  y_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False)

# ------------------------
# 7. 网络结构
# ------------------------
class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_small = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_large = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(out_channels * 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        small = self.conv_small(x)
        large = self.conv_large(x)
        x = torch.cat([small, large], dim=1)
        x = self.bn(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        return x

class AttentionFusion(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, 2),
            nn.Softmax(dim=-1)
        )
    def forward(self, var_feat, beta_feat):
        weights = self.attn(torch.cat([var_feat, beta_feat], dim=-1))
        w_var = weights[:,0].unsqueeze(-1)
        w_beta= weights[:,1].unsqueeze(-1)
        fusion = w_var * var_feat + w_beta * beta_feat
        return fusion

class DualLSTM_Attention(nn.Module):
    def __init__(self, var_dim, beta_dim, hidden_dim=64):
        super().__init__()
        self.cnn_var = MultiScaleCNN(var_dim, out_channels=16)
        self.cnn_beta = MultiScaleCNN(beta_dim, out_channels=16)

        self.var_lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, batch_first=True)
        self.beta_lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, batch_first=True)

        self.attn = AttentionFusion(hidden_dim * 2)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, x_var, x_beta, coords):
        x_var = self.cnn_var(x_var)
        x_beta= self.cnn_beta(x_beta)

        var_out, _ = self.var_lstm(x_var)
        beta_out, _ = self.beta_lstm(x_beta)

        var_last = var_out[:,-1,:]
        beta_last= beta_out[:,-1,:]

        fusion = self.attn(var_last, beta_last)
        fusion = torch.cat([fusion, coords], dim=-1)

        out = self.fc(fusion)
        return out

# ------------------------
# 8. 损失函数
# ------------------------
class WeightedMSELoss(nn.Module):
    def __init__(self, weight=5.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred, target):
        loss = (pred - target)**2
        weights = torch.where(torch.abs(target) < 10, self.weight, 1.0)
        weighted_loss = loss * weights
        return weighted_loss.mean()

# ------------------------
# 9. 训练模型
# ------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DualLSTM_Attention(var_dim=9, beta_dim=9, hidden_dim=64).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
loss_fn = WeightedMSELoss(weight=5.0)

for epoch in range(1500):
    model.train()
    train_losses = []
    for X_var, X_beta, X_coord, yb in train_loader:
        X_var, X_beta, X_coord, yb = X_var.to(device), X_beta.to(device), X_coord.to(device), yb.to(device)

        pred = model(X_var, X_beta, coords=X_coord.mean(dim=1))
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_losses.append(loss.item())

    avg_loss = np.mean(train_losses)
    scheduler.step(avg_loss)
    print(f"Epoch {epoch+1}  Train Loss = {avg_loss:.4f}")

# ------------------------
# 10. 测试集上预测
# ------------------------
model.eval()
preds = []
trues = []

with torch.no_grad():
    for X_var, X_beta, X_coord, yb in test_loader:
        X_var, X_beta, X_coord, yb = X_var.to(device), X_beta.to(device), X_coord.to(device), yb.to(device)
        pred = model(X_var, X_beta, coords=X_coord.mean(dim=1))
        preds.append(pred.cpu().numpy())
        trues.append(yb.cpu().numpy())

preds = np.concatenate(preds, axis=0)
trues = np.concatenate(trues, axis=0)

# ------------------------
# 11. 输出最终指标
# ------------------------
r2 = r2_score(trues, preds)
mse = mean_squared_error(trues, preds) # 不加 squared 参数
rmse = np.sqrt(mse)
mae = mean_absolute_error(trues, preds)

print(f"\n测试集结果:  R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")