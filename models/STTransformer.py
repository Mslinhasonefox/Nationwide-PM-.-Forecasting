import os
import numpy as np
import torch
from click.core import batch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


class PollutionDataset(Dataset):
    def __init__(self, root_dir, years, time_steps, max_stations=9):
        """
        root_dir: 数据根目录 (e.g., "D:/")
        years: 年份列表 (e.g., [2017, 2018, ..., 2022])
        time_steps: 时间步长 (e.g., 7 days sliding window)
        max_stations: 固定的最大站点数，用于对齐
        """
        self.data = []
        self.labels = []
        self.max_stations = max_stations

        for year in years:
            year_dir = os.path.join(root_dir, str(year))
            daily_files = sorted([os.path.join(year_dir, f) for f in os.listdir(year_dir) if f.endswith('.txt')])

            for i in range(0, len(daily_files) - time_steps + 1):  # 滑动窗口
                temporal_data = []
                for j in range(time_steps):
                    # 提取需要的特征列
                    daily_data = np.loadtxt(
                        daily_files[i + j],
                        delimiter=',',
                        usecols=range(6, 15)  # 从第 1 列（Longitude）到第 15 列（NDVI）
                    )

                    # 对齐 num_stations
                    if daily_data.shape[0] < self.max_stations:  # 补零填充
                        padding = np.zeros((self.max_stations - daily_data.shape[0], daily_data.shape[1]))
                        daily_data = np.vstack([daily_data, padding])
                    elif daily_data.shape[0] > self.max_stations:  # 截断
                        daily_data = daily_data[:self.max_stations, :]

                    temporal_data.append(daily_data)

                # 堆叠时间步数据，确保四维结构
                temporal_data = np.stack(temporal_data, axis=0)  # Shape: [time_steps, max_stations, num_features]
                temporal_data = temporal_data.transpose(1, 0, 2)  # Shape: [max_stations, time_steps, num_features]
                self.data.append(temporal_data)

                # 提取 PM2.5 作为标签（最后一天）
                daily_pm = np.loadtxt(
                    daily_files[i + time_steps - 1],
                    delimiter=',',
                    usecols=-1  # 最后一列是 PM2.5
                )
                # 对齐 PM2.5 标签
                if daily_pm.shape[0] < self.max_stations:  # 补零填充
                    padding = np.zeros(self.max_stations - daily_pm.shape[0])
                    daily_pm = np.concatenate([daily_pm, padding])
                elif daily_pm.shape[0] > self.max_stations:  # 截断
                    daily_pm = daily_pm[:self.max_stations]

                self.labels.append(daily_pm)

        self.data = np.array(self.data)  # Shape: [samples, max_stations, time_steps, num_features]
        self.labels = np.array(self.labels)  # Shape: [samples, max_stations]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)


# Embedding Modules
# =======================
class SpatialEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(SpatialEmbedding, self).__init__()
        self.fc = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        return self.fc(x)


class TemporalEmbedding(nn.Module):
    def __init__(self, embed_dim, time_steps):
        super(TemporalEmbedding, self).__init__()
        self.positional_encoding = nn.Parameter(torch.randn(1, time_steps, embed_dim))

    def forward(self, x):
        return x + self.positional_encoding


class SSelfAttention(nn.Module):
    def __init__(self, embed_dim, heads):
        super(SSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        assert self.head_dim * heads == embed_dim, "Embedding size must be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_dim)

    def forward(self, values, keys, query):
        # 获取输入的形状
        N, T, S, C = query.shape

        # 重塑形状
        values = values.view(N, T, S, self.heads, self.head_dim)
        keys = keys.view(N, T, S, self.heads, self.head_dim)
        query = query.view(N, T, S, self.heads, self.head_dim)

        # 通过线性层
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        # 计算 attention
        energy = torch.einsum("ntshd,ntkhd->ntskh", queries, keys) / (self.embed_dim ** 0.5)

        attention = torch.softmax(energy, dim=-1)

        out = torch.einsum("ntskh,ntshd->ntshd", attention, values).reshape(N, T, S, self.heads * self.head_dim)

        out = self.fc_out(out)

        return out


class TSelfAttention(nn.Module):
    def __init__(self, embed_dim, heads):
        super(TSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        assert self.head_dim * heads == embed_dim, "Embedding size must be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_dim)

    def forward(self, values, keys, query):
        N, T, S, C = query.shape
        values = values.view(N, T, S, self.heads, self.head_dim)
        keys = keys.view(N, T, S, self.heads, self.head_dim)
        query = query.view(N, T, S, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        energy = torch.einsum("ntshd,ntkhd->ntksh", queries, keys) / (self.embed_dim ** 0.5)
        attention = torch.softmax(energy, dim=-1)
        out = torch.einsum("ntksh,ntshd->ntshd", attention, values).reshape(N, T, S, self.heads * self.head_dim)

        return self.fc_out(out)


###主函数
class STTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, dropout=0.1):
        """
        STTransformer 模型
        input_dim: 输入特征的维度
        embed_dim: 嵌入维度
        num_heads: 多头注意力机制的头数
        num_layers: 注意力层的堆叠数
        dropout: Dropout 概率
        """
        super(STTransformer, self).__init__()
        self.spatial_embed = SpatialEmbedding(input_dim, embed_dim)
        self.temporal_embed = TemporalEmbedding(embed_dim, time_steps=7)
        self.spatial_attention = nn.ModuleList([SSelfAttention(embed_dim, num_heads) for _ in range(num_layers)])
        self.temporal_attention = nn.ModuleList([TSelfAttention(embed_dim, num_heads) for _ in range(num_layers)])

        self.fc_out = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.ReLU()  # 保证输出非负
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 空间嵌入
        x_spatial = self.spatial_embed(x)

        # 时间嵌入
        x_temporal = self.temporal_embed(x_spatial)

        # 调整形状为 [batch, time, stations, embed_dim]
        x_temporal = x_temporal.permute(0, 2, 1, 3)

        # 逐层应用空间和时间注意力
        for s_attention, t_attention in zip(self.spatial_attention, self.temporal_attention):
            x_temporal = s_attention(x_temporal, x_temporal, x_temporal)
            x_temporal = t_attention(x_temporal, x_temporal, x_temporal)

        # 最终输出映射
        x_out = self.fc_out(x_temporal)
        return x_out


# 定义训练函数
# =======================
def train_model(model, train_loader, criterion, optimizer, epochs=999, device='cpu'):
    model = model.to(device)  # 将模型移动到设备（CPU 或 GPU）
    for epoch in range(epochs):
        model.train()  # 设置模型为训练模式
        epoch_loss = 0

        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)  # 将数据移动到设备

            # 前向传播
            optimizer.zero_grad()
            outputs = model(data)

            print(f"Outputs: min={outputs.min()}, max={outputs.max()}, mean={outputs.mean()}")
            print(f"Labels: min={labels.min()}, max={labels.max()}, mean={labels.mean()}")
            # 计算损失
            outputs = outputs[:, -1, :, 0]
            loss = criterion(outputs.squeeze(-1), labels)
            epoch_loss += loss.item()

            # 反向传播
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

        model.eval()  # 设置模型为评估模式
        val_loss = 0
        all_preds = []
        all_labels = []
        val_mse, val_mae, val_mape, val_r2 = 0, 0, 0, 0

        with torch.no_grad():  # 禁用梯度计算
            for batch_idx, (data, labels) in enumerate(val_loader):
                data, labels = data.to(device), labels.to(device)  # 将数据移动到设备

                # 前向传播
                outputs = model(data)

                # 提取最后一个时间步的预测值
                outputs = outputs[:, -1, :, 0]
                loss = criterion(outputs.squeeze(-1), labels)
                val_loss += loss.item()
                preds = outputs.squeeze(-1).cpu().numpy()  # Shape: [batch_size, num_stations]
                labels = labels.cpu().numpy()

                # 存储预测值和真实值
                all_preds.append(preds)
                all_labels.append(labels)

        val_all_preds = np.vstack(all_preds)  # Shape: [num_samples, num_stations]
        val_all_labels = np.vstack(all_labels)  # Shape: [num_samples, num_stations]

        # 将多站点的预测值和真实值展平成一维数组
        val_all_preds_flat = val_all_preds.flatten()
        val_all_labels_flat = val_all_labels.flatten()
        val_mae = mean_absolute_error(val_all_preds_flat, val_all_labels_flat)
        val_mse = mean_squared_error(val_all_labels_flat, val_all_preds_flat)
        val_mape = np.mean(np.abs((val_all_labels_flat - val_all_preds_flat) / val_all_labels_flat)) * 100  # 处理百分比误差
        val_r2 = r2_score(val_all_labels_flat, val_all_preds_flat)

        metrics = {
            "MAE": val_mae,
            "MSE": val_mse,
            "MAPE": val_mape,
            "R2": val_r2
        }

        print(f"Val Metrics:{val_mae, val_mse, val_mape, val_r2}")

        # 输出验证损失
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss / len(val_loader):.4f}")


# =======================
# 数据加载
# =======================
# 使用2017年的数据集
train_dataset = PollutionDataset(
    root_dir="/content/drive/MyDrive/mmnewnew",
    years=[2017, 2018, 2019, 2020],  # 使用2017年的数据
    time_steps=7,
    max_stations=9
)

val_dataset = PollutionDataset(
    root_dir="/content/drive/MyDrive/mmnewnew",
    years=[2021],  # 使用2017年的数据
    time_steps=7,
    max_stations=9
)

test_dataset = PollutionDataset(
    root_dir="/content/drive/MyDrive/mmnewnew",
    years=[2022],  # 使用2017年的数据
    time_steps=7,
    max_stations=9
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, drop_last=True)
# =======================
# 模型和优化器
# =======================
# 定义模型
model = STTransformer(input_dim=9, embed_dim=128, num_heads=8, num_layers=2, dropout=0.1)

# 定义损失函数
criterion = nn.HuberLoss(delta=0.1)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# =======================
# 训练模型
# =======================
train_model(
    model=model,
    train_loader=train_loader,
    criterion=criterion,
    optimizer=optimizer,
    epochs=51,  # 训练5个轮次
    device='cuda' if torch.cuda.is_available() else 'cpu'
)


def test_model(model, test_loader, device):
    """
    测试模型并计算评估指标（MAE、MSE、MAPE 和 R²）。
    Args:
        model: 训练好的模型。
        test_loader: 测试集 DataLoader。
        device: 设备 ('cuda' 或 'cpu')。
    Returns:
        metrics: 包含评估指标的字典。
    """
    model.eval()  # 切换到评估模式
    all_preds = []
    all_labels = []

    with torch.no_grad():  # 测试时不需要梯度
        for batch_idx, (data, labels) in enumerate(test_loader):
            data, labels = data.to(device), labels.to(device)

            # 前向传播
            outputs = model(data)
            outputs = outputs[:, -1, :, 0]

            # 如果输出有额外的维度，移除（只保留预测值和标签的形状匹配）
            preds = outputs.squeeze(-1).cpu().numpy()  # Shape: [batch_size, num_stations]
            labels = labels.cpu().numpy()

            # 存储预测值和真实值
            all_preds.append(preds)
            all_labels.append(labels)

    # 将所有批次的预测值和真实值拼接
    all_preds = np.vstack(all_preds)  # Shape: [num_samples, num_stations]
    all_labels = np.vstack(all_labels)  # Shape: [num_samples, num_stations]

    # 将多站点的预测值和真实值展平成一维数组
    all_preds_flat = all_preds.flatten()
    all_labels_flat = all_labels.flatten()

    # 计算评估指标
    mae = mean_absolute_error(all_labels_flat, all_preds_flat)
    mse = mean_squared_error(all_labels_flat, all_preds_flat)
    mape = np.mean(np.abs((all_labels_flat - all_preds_flat) / all_labels_flat)) * 100  # 处理百分比误差
    r2 = r2_score(all_labels_flat, all_preds_flat)

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "MAPE": mape,
        "R2": r2
    }

    print(f"Test Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    return metrics, all_labels_flat, all_preds_flat


metrics, all_labels_flat, all_preds_flat = test_model(
    model=model,
    test_loader=test_loader,  # 训练5个轮次
    device='cuda' if torch.cuda.is_available() else 'cpu'
)


def plot_time_series(all_labels_flat, all_preds_flat, title="ST Transformers Predictions vs True Values", threshold=250,
                     start_idx=0, end_idx=None):
    """
    可视化预测值与真实值的对比
    """
    if end_idx is None:
        end_idx = len(all_labels_flat)

    # 对数据进行过滤：仅保留 <= threshold 的数据
    mask = (all_labels_flat <= threshold) & (all_preds_flat <= threshold)
    true_values_filtered = all_labels_flat[mask]
    predicted_values_filtered = all_preds_flat[mask]

    # 限制显示范围
    all_labels__filtered = true_values_filtered[start_idx:end_idx]
    all_preds_filtered = predicted_values_filtered[start_idx:end_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(all_labels__filtered, label="True Values", color="blue", alpha=0.7)  # 绘制真实值折线图
    plt.plot(all_preds_filtered, label="Predicted Values", color="red", alpha=0.7)  # 绘制预测值折线图
    plt.legend()
    plt.xlabel("Time Step")
    plt.ylabel("PM2.5 Values")
    plt.title(title)
    plt.grid()
    plt.show()


# 可视化测试集结果
plot_time_series(all_labels_flat, all_preds_flat, title="CNN-LSTM Predictions vs True Values")
