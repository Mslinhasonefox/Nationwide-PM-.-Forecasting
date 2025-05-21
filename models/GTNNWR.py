# gtnnwr按站点划分数据
# ------------------------------------------------------
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------- 0. 读 3 份已分好的 csv ----------
csv_train = "/content/drive/MyDrive/split821/pm25_train.csv"
csv_val   = "/content/drive/MyDrive/split821/pm25_val.csv"
csv_test  = "/content/drive/MyDrive/split821/pm25_test.csv"

df_tr  = pd.read_csv(csv_train)
df_val = pd.read_csv(csv_val)
df_te  = pd.read_csv(csv_test)

# 把它们竖向拼回去（方便按行写 β）
df_all = (pd.concat([df_tr, df_val, df_te], axis=0)
            .reset_index(drop=True))

# 记录各自的行号索引，后面回填 β 时要用
idx_train = np.arange(len(df_tr))
idx_val   = np.arange(len(df_tr), len(df_tr)+len(df_val))
idx_test  = np.arange(len(df_tr)+len(df_val), len(df_all))

# ---------- 1. 提取特征 ----------
feature_cols = ["AOD","Surface_Pressure","Temperature","Wind_Speed",
                "Wind_Direction","Relative_Humidity",
                "Planetary_Boundary_Height","NDVI","DEM"]

coords = df_all[["longitude","latitude"]].values
date_num = (pd.to_datetime(df_all[["year","month","day"]]) -
            pd.to_datetime(df_all[["year","month","day"]]).min()
           ).dt.days.values.astype(float)

# ---------- 2. 标准化 ----------
scX, scY = StandardScaler(), StandardScaler()
X_all = torch.tensor(scX.fit_transform(df_all[feature_cols]),
                     dtype=torch.float32)
y_all = torch.tensor(scY.fit_transform(df_all[["PM25"]]),
                     dtype=torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========== 3. 只用 *train* 样本做邻接矩阵 ===========
def build_dist(idx_row, idx_col):
    dS = np.sqrt((coords[idx_row,None,0]-coords[idx_col,None,0].T)**2 +
                 (coords[idx_row,None,1]-coords[idx_col,None,1].T)**2)
    dT = np.abs(date_num[idx_row,None]-date_num[idx_col,None].T)
    # 全局 min/max 归一化
    return ((dS-dS.min())/(np.ptp(dS)+1e-6),
            (dT-dT.min())/(np.ptp(dT)+1e-6))

dS_tr,dT_tr = build_dist(idx_train, idx_train)        # (n_tr,n_tr)
dS_val,dT_val = build_dist(idx_val,  idx_train)       # (n_val,n_tr)
dS_te,dT_te   = build_dist(idx_test, idx_train)       # (n_te,n_tr)

toT = lambda a: torch.tensor(a[...,None],dtype=torch.float32,device=device)
D_s_tr,D_t_tr = toT(dS_tr),toT(dT_tr)
D_s_val,D_t_val = toT(dS_val),toT(dT_val)
D_s_te,D_t_te   = toT(dS_te),toT(dT_te)

X_tr,y_tr = X_all[idx_train].to(device), y_all[idx_train].to(device)
X_val,y_val = X_all[idx_val].to(device), y_all[idx_val].to(device)

# ---------- 4. ↓↓↓ 你的 GTNNWR 网络结构 & 训练代码保持不变 ↓↓↓ ----------
# 直接复制你那块 4,5,6 的代码即可
# （为节省篇幅，此处省略，记得把 loss_fn / early-stop 那段粘进来）
# ---------------- 4. 网络结构（缩小+Dropout） ----------------
def mlp(in_f,out_f):
    return nn.Sequential(nn.Linear(in_f,out_f),
                         nn.ReLU(),
                         nn.Dropout(0.2))

class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(mlp(1,16), mlp(16,8))
    def forward(self,x): return self.mlp(x)

class TNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(mlp(1,16), mlp(16,8))
    def forward(self,x): return self.mlp(x)

class STNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(mlp(16,8), nn.Linear(8,1), nn.Softplus())
    def forward(self,x): return self.mlp(x).squeeze(-1)

class GTNNWR(nn.Module):
    def __init__(self):
        super().__init__()
        self.snn = SNN()     # ← 这里正确初始化
        self.tnn = TNN()
        self.stnn = STNN()

    def forward(self,D_s,D_t):
        Fs, Ft = self.snn(D_s), self.tnn(D_t)
        return self.stnn(torch.cat([Fs,Ft],-1))

# ---------------- 5. 训练 ----------------
model = GTNNWR().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()

epochs, patience, best_val = 1000, 60, 1e9
loss_hist, val_hist = [], []

for ep in range(epochs):
    # --- train ---
    model.train(); optimizer.zero_grad()
    W = model(D_s_tr,D_t_tr)
    Wn = W/(W.sum(1,keepdim=True)+1e-6)
    y_hat = Wn @ y_tr
    loss = loss_fn(y_hat, y_tr); loss.backward(); optimizer.step()
    # --- val ---
    model.eval()
    with torch.no_grad():
        Wv = model(D_s_val,D_t_val)
        Wnv= Wv/(Wv.sum(1,keepdim=True)+1e-6)
        y_val_hat = Wnv @ y_tr      # 仍用训练 y 做加权
        val_loss = loss_fn(y_val_hat, y_val).item()
    loss_hist.append(loss.item()); val_hist.append(val_loss)
    # early‑stop
    if val_loss < best_val-1e-4:
        best_val, patience_cnt = val_loss, 0
        best_state = model.state_dict()
    else:
        patience_cnt += 1
    if patience_cnt > patience: break

print(f"Stop@{ep+1}, best val={best_val:.4f}")
model.load_state_dict(best_state)

def gtwr_beta(W, X_nei, y_nei):
    n, m = W.shape; p = X_nei.shape[1]
    beta = torch.zeros(n,p,device=device); I=torch.eye(p,device=device)
    for i in range(n):
        Wi = torch.diag(W[i])
        beta[i]=torch.linalg.solve(X_nei.T@Wi@X_nei+1e-4*I,X_nei.T@Wi@y_nei).flatten()
    return beta

with torch.no_grad():
    W_te = model(D_s_te, D_t_te)
    Wn_te= W_te/(W_te.sum(1,keepdim=True)+1e-6)
    y_te_hat = (Wn_te @ y_tr).cpu().numpy()
    y_te_hat = scY.inverse_transform(y_te_hat)         # 反标准化
    y_te_true= scY.inverse_transform(y_all[idx_test])      # 反标准化

# 指标
r2 = r2_score(y_te_true, y_te_hat)
mse= mean_squared_error(y_te_true, y_te_hat)
rmse= np.sqrt(mse)
mae= mean_absolute_error(y_te_true, y_te_hat)
mape= np.mean(np.abs((y_te_true-y_te_hat)/(y_te_true+1e-6)))*100
print(f"Test  R²={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}  MAPE={mape:.2f}%")

# -------------------------------------------------------------------
# 假设最后已得到 model, best_state，并 load_state_dict(best_state)

# ---------- 5. 预测 & 生成 β ----------
@torch.no_grad()
def predict_beta(D_s,D_t):
    W   = model(D_s,D_t)             # (n_row,n_tr)
    Wn  = W/(W.sum(1,keepdim=True)+1e-6)
    ŷ   = Wn @ y_tr                 # (n_row,1)
    # 求 β
    n,p = W.shape[0], X_tr.shape[1]
    beta = torch.zeros(n,p,device=device)
    I = torch.eye(p,device=device)
    for i in range(n):
        Wi = torch.diag(Wn[i])
        beta[i] = torch.linalg.solve(
                    X_tr.T@Wi@X_tr + 1e-4*I,
                    X_tr.T@Wi@y_tr).flatten()
    return ŷ.cpu(), beta.cpu()

y_hat_tr, β_tr = predict_beta(D_s_tr,D_t_tr)
y_hat_val,β_val= predict_beta(D_s_val,D_t_val)
y_hat_te, β_te = predict_beta(D_s_te,D_t_te)

# ---------- 6. 反标准化 & 指标 ----------
def inv(a): return scY.inverse_transform(a)
print("VAL  R2=%.3f"%
      r2_score(inv(y_val.cpu()), inv(y_hat_val)))
print("TEST R2=%.3f"%
      r2_score(inv(y_all[idx_test]), inv(y_hat_te)))

# ---------- 7. 回填 β & 保存 ----------
N,p = len(df_all), len(feature_cols)
β_all = np.zeros((N,p))
β_all[idx_train] = β_tr
β_all[idx_val]   = β_val
β_all[idx_test]  = β_te

df_out = pd.concat([df_all.reset_index(drop=True),
                    pd.DataFrame(β_all,columns=[f"beta_{c}" for c in feature_cols])],
                   axis=1)
df_out.to_csv("gtnnwr_full_betagood.csv", index=False, encoding="utf-8-sig")
print("✔ 已保存 gtnnwr_full_betagood.csv")
