# Pipeline

下面把 SGM-Net 从“网络结构 → 表示生成 → 训练目标 → 检索流程（离线/在线）”讲清楚，并在关键处给出张量形状与计算细节，便于实现或核对。

## 1) 输入与骨干特征

* **输入**：RGB，通常 3×224×224（论文训练默认）。
* **Backbone**：VGG-16 或 ResNet-50 的**前四个 stage/block**（Conv1–4）。输出特征记为
  (X\in\mathbb{R}^{B\times C\times H\times W})，典型为 (C!=!256,,H!=!W!=!14)。
* **Non-local block（点积型）**：接在 backbone 之后，用于**全局上下文建模**（长程依赖）。前后通常有通道自适配的 1×1 Conv 与池化。输出记为
  (X_g\in\mathbb{R}^{B\times C\times H\times W})（与 (X) 同尺度）。

> 这一步“非局部注意”让特征在大视角差、尺度差时仍能对齐语义。

## 2) 双分支表示：全局 Ω 与半全局 Φ

SGM-Net 输出两个互补的向量：

## 2.1 全局描述子 Ω（粗召回）

* **做法**：直接**展平** Non-local 输出：
  (\displaystyle \Omega=\mathrm{Flatten}(X_g)\in\mathbb{R}^{B\times (C!\cdot!H!\cdot!W)})。
  例如 (256\times14\times14 \Rightarrow 50176) 维。
* **用途**：用于**大规模库**的快速相似度检索（点积/余弦），得到候选 Top-N。

### 2.2 CSMG 半全局描述子 Φ（精匹配）

这是论文的关键改造：在 NetVLAD 思路上**保留结构**而非仅做全局汇聚。

#### (a) 聚类中心与相似度掩膜

* 设有 (K) 个聚类中心 ({c_k}_{k=1}^K)，每个 (c_k\in\mathbb{R}^{C})（可学习）。
* 把空间像素 ((H!\times!W)) 展平为 (N) 个局部特征向量 ({x_i}_{i=1}^N)，每个 (x_i\in\mathbb{R}^C)。
* 计算**相似度**（点积或余弦），并做 ReLU 以抑制负相似：
  [
  s_{k,i}=\mathrm{ReLU}(c_k^\top x_i),\quad
  X^{*}*{k,i}=s*{k,i}\cdot x_i
  ]
  直观上，(X^{*}_{k,\cdot}) 是“**被第 k 类激活的像素**”的加权特征。

#### (b) 共识特征（节点）与空间分布

* **节点特征**（第 k 个“语义节点”）：
  (\displaystyle d_k=\sum_{i=1}^{N} X^{*}_{k,i}\in\mathbb{R}^{C})。
* **节点覆盖度/面积**：(\displaystyle a_k=\sum_{i=1}^{N}\mathbf{1}[s_{k,i}>0]) 或用 (s_{k,i}) 的和做软面积。
* **节点位置**（可选）：用 ({s_{k,i}}) 对像素坐标求加权平均，可得每个节点在图像平面中的“语义重心”。

> 相比 NetVLAD 的“残差累加”，这里是**相似度掩膜聚合**，并且**显式保存每个簇的空间覆盖**，从而对旋转/俯仰等视角变化更稳健。

#### (c) 节点排序与向量化

* 按节点的**覆盖面积 (a_k)** 从大到小排序，得到序列 ((d_{(1)},\dots,d_{(K)}))。
* **拼接展平**：(\displaystyle \Phi=\mathrm{Concat}\big(d_{(1)},\dots,d_{(K)}\big)\in\mathbb{R}^{B\times (K\cdot C)})。
  典型 (C!=!256\Rightarrow \mathrm{dim}(\Phi)=K\times256)（如 (K!=!4\Rightarrow 1024) 维）。

> “面积排序”是把“结构（大块/小块）关系”固化到向量序中，便于后续用**普通向量相似度**完成结构对齐。

## 3) 训练目标（联合优化）

总体损失为两部分的加权和：
[
L=\beta\cdot L_{\text{global}}^{\Omega}+(1-\beta)\cdot L_{\text{triplet}}^{\Phi},\quad \beta\approx 0.5
]

* **全局分支（Ω）**：余弦嵌入损失（CosineEmbeddingLoss）。
  给定 (anchor, positive, negative)，使 (\cos(\Omega_a,\Omega_p)\to 1)，(\cos(\Omega_a,\Omega_n)\to -1)。
* **半全局分支（Φ）**：**余弦三元组损失**（Cosine Triplet）。
  以余弦相似为度量，鼓励 (\cos(\Phi_a,\Phi_p)) 大、(\cos(\Phi_a,\Phi_n)) 小。
  典型 margin 0.2；anchor 通常来自**卫星**，正/负来自 **UAV**。

> 这样训练出的两个表征分工明确：Ω 负责**粗召回**，Φ 负责**结构敏感的精匹配**。

## 4) 检索流程（推荐的分层检索）

SGM-Net的推理/检索是**两阶段**的：

### 4.1 离线建库（一次性/周期性）

对**参考地图库**（卫星/正射/重建影像等）逐张提取并缓存：

* **全局向量** (\Omega^r)（高维，召回用）
* **半全局向量** (\Phi^r)（低维，重排用）
  可分别存两套索引：
* 全库的 ({\Omega^r})：用 FAISS/内积做**快速召回**；
* 子库候选的 ({\Phi^r})：用余弦相似做**精排序**。

### 4.2 在线查询（实时）

对 UAV 查询图像 (q)：

1. **提特征**：得到 (\Omega^q) 与 (\Phi^q)。
2. **粗召回（Top-N）**：对全库的 ({\Omega^r}) 做内积/余弦相似，取 Top-N 候选 (\mathcal{C}(q))。
3. **精重排**：在 (\mathcal{C}(q)) 上，用 (\Phi) 的**余弦相似**重新排序，输出最终 Top-K。

   > 若需要更强的结构一致性检验，可在 Top-M（更小）上做**节点级（d_k）对齐/一致性度量**作为 re-score。

> 这种“Ω 召回 + Φ 重排”的层级策略，既扛视角差又高效：Ω 粗而全、Φ 准而轻。

## 5) 关键超参与形状核对

* **通道数 (C)**：通常 256（来自 backbone 的 stage4 + 通道适配）。
* **空间尺寸 (H!\times!W)**：常见 14×14（224 输入，步长 16）。
* **聚类数 (K)**：4–6 为常用折中（维度/速度/效果平衡）。
* **维度**：(\dim(\Omega)=C\cdot H\cdot W)，(\dim(\Phi)=K\cdot C)。
* **损失权重 (\beta)**：0.5。**优化器**：Adam(lr≈1e-3)。**输入**：224×224。

## 6) 与 NetVLAD 的本质差异（一句话）

* **NetVLAD**：对每簇做“残差累加 → 拼接”，得到**全局**集合表示，**空间位置信息被抹平**。
* **SGM-Net/CSMG**：对每簇做“相似度掩膜聚合”，并**保留节点的空间覆盖/排序**，把“**结构**”显式编码进半全局向量 Φ；再与全局 Ω 组合实现**粗到细检索**。

## 7) 最小实现轮廓（与 `nets.py` 对齐思路）

```python
# x: Bx3x224x224
X = backbone_conv1_4(x)                  # -> BxC(256)xHxW(14)
Xg = non_local_block(X)                  # -> BxC×H×W

# Ω: 全局
Omega = Xg.flatten(1)                    # Bx(CHW)

# Φ: CSMG
# C: KxC 为聚类中心；把 X 展平成 BxN(=HW)xC
X_flat = X.flatten(2).transpose(1, 2)    # BxN×C
C_ = C.t().unsqueeze(0).expand(B, -1, -1)  # BxC×K
S = torch.bmm(X_flat, C_)                # BxN×K  ~ x_i^T c_k
S = F.relu(S).transpose(1, 2)            # BxK×N

# 掩膜聚合
X_star = S.unsqueeze(3) * X_flat.unsqueeze(1)  # BxK×N×C
d = X_star.sum(dim=2)                           # BxK×C

# 面积排序（按 S 的和或计数）
area = S.sum(dim=2)                             # BxK
idx = torch.argsort(area, dim=1, descending=True)
d_sorted = torch.gather(d, 1, idx.unsqueeze(-1).expand_as(d))  # BxK×C

Phi = d_sorted.flatten(1)                      # Bx(KC)
return Omega, Phi
```

---

**一句话记忆**：SGM-Net 用 Non-local 提升全局语义（Ω），再用“相似度掩膜 + 面积排序”把语义结构固化到半全局向量（Φ），最后以“Ω 召回、Φ 重排”的两阶段流程完成稳健且高效的跨视角检索。
