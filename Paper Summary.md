# Paper Summary.md

## 论文信息（用于复现定位）

* 标题：**A Scene Graph Encoding and Matching Network for UAV Visual Localization (SGM-Net)**
* 代码仓库（仅推理与网络结构）：`rduan036/scene-graph-matching-demo`（论文中也给出该地址）

---

## 任务与总体思路

* **任务**：在多源（UAV/卫星/点云）+跨视角条件下进行视觉地点检索（VPR），用查询的UAV视图在参考地图中找到最佳匹配位置。
* **核心思路**：将图像编码为**双层次表示**：

  1. 来自**Non-local**模块的**全局描述子**（用于快速候选召回）；
  2. 由**CSMG（Cluster Similarity Masking Graph）**编码器产生的**结构化半全局（场景图）描述子**（用于精匹配）。两者**层级式粗到细**检索。 

---

## 网络结构（实现要点）

* **Backbone**：预训练 **VGG-16 或 ResNet-50 的前四个 block**。作者将 **Conv5** 用 **Non-local Block** 替换，并在 Non-local 前后加**通道自适应Conv**与**MaxPool**。 
* **全局分支**：Non-local 输出张量 (x_g \in \mathbb{R}^{256\times14\times14})，**直接展平**为全局描述子 (\Omega)。
* **CSMG 半全局分支**（NetVLAD改造）：

  * 不再求“特征−聚类中心残差”之和（NetVLAD），而是对与聚类中心相似的特征做**相似度加权聚合**，得到每簇的**共识特征**与**节点描述子 (d_k)**；同时恢复其在图像平面上的**空间分布**并**按覆盖面积排序后再展平**，形成结构感知的扁平向量。  
  * **关键公式**：

    * 相似度掩膜：(X^{*}(k,i)=\mathrm{ReLU}(c_k^\top x_i)\odot x_i)。
    * 节点描述子：(d_k=\sum_i X^{*}(k,i))。
    * 全局描述子：(\Omega=\mathrm{Flatten}(x_g))。
  * **实现提示**：相似度张量 (S\in\mathbb{R}^{B\times K\times N}) 可用 `torch.bmm` 计算；用 ReLU 去除非正相似度，保证端到端可导；随后与特征 X 做按元素乘积得到掩膜特征。
* **描述子维度**：每个半全局节点的维度 **D = 256**（沿用 NetVLAD 推荐），故**最终半全局向量维度 = K × 256**。

---

## 训练目标与采样

* **损失函数**：提出**余弦三元组损失**（将 NetVLAD 的 L2 距离替换为 **cosine**），并与**全局分支的余弦嵌入损失**加权求和：
  [
  L=\beta,L^{\Omega}*{\text{cos}} + (1-\beta),L*{\text{cos}},\quad \beta=0.5\ (\text{默认})
  ]
  用于**同时训练全局与半全局**两条分支。 
* **三元组构造**：Anchor 来自**卫星图**，正/负样本来自**UAV图**；弱监督排序损失思想沿用 NetVLAD。 

---

## 数据与评测协议（再现实验所需）

* **数据集**：

  * **University-1652**：1652所大学建筑，训练集含701处；卫星正射+合成UAV多角度视图（GE 3D引擎生成）。
  * **ALTO**：13,781张高空航摄+3,742张卫星图，主要是自然场景，时空变化大。
  * **SenseFly**：443张UAV图与由LiDAR+RGB重建的全局彩色点云正射图，用于**零样本泛化测试**。
* **查询/库设置**：均以**UAV图为查询**，匹配**预先编码的参考地图**；指标为 **R@1 / R@5**。
* **层级检索流程（推理时）**：

  1. 用(\Omega)对整库做**点积相似度**得到候选Top-N；
  2. 对候选再用**场景图半全局描述子**做**余弦相似度**排序。论文给出了**Algorithm 1**伪代码。

---

## 训练食谱（可直接落地）

> 下表为作者在 **RTX3090** 上的训练设置，适合作为复现起点。

* **输入尺寸**：`3×224×224`；**Batch**：`128`；**Epoch**：`100`；**优化器/学习率**：`Adam / 1e-3`；**输出维度**：`K×256`（半全局）；**增强**：`resize, random crop, padding`。
* **聚类数 K 的选择**：

  * **K=4** 在 100 epoch 内常取得较优R@1；**K=6** 最高但更大更慢；综合考虑，**K≈4** 是效率/精度的**合理折中**。 
* **收敛与过拟合提示**：80个epoch后训练损失下降趋缓（≈每epoch 0.001），可视作收敛；数据存在合成偏差，**K过大时（如16）易在≈80 epoch过拟合**。 

---

## 复杂度与部署（实现时的工程考量）

* 以 **VGG-16+K=4** 为例：参数量 **8.9M**（对比 NetVLAD 140.6M），FLOPs **15.0G**（NetVLAD 94.2G）。在 **RTX3090 ≈ 300 FPS**，**Jetson Xavier NX ≈ 37 FPS**。 

---

## 关键实现清单（你需要自己补的训练部分）

1. **数据管线**

   * 读取 **UAV–Sat** 对；按论文协议构造 **(anchor=Sat, positive/negative=UAV)** 三元组/矿难样本。
   * 预处理/增强：**resize → random crop → padding**；**224×224** 输入张量。
2. **模型前向**

   * Backbone 前四个block → **Non-local**（含通道适配Conv+MaxPool）并**分岔**：

     * **全局**：`Flatten(NonLocalOut)` → (\Omega)。
     * **半全局**：将同一特征图送入 **CSMG**：用 **`torch.bmm`** 计算 (S=c^\top x)，**ReLU** 掩膜 → 聚合得到 (\Phi=[d_1,\dots,d_K]) → **按节点覆盖面积排序并展平**。 
3. **损失**

   * **Cosine Triplet**（半全局向量 (f)）+ **Global Cosine Embedding**（(\Omega)）；**(\beta=0.5)**。
4. **优化/调参**

   * **Adam(lr=1e-3), batch=128, epoch≈100**；K从 ([4,6]) 试起；关注80 epoch 后收敛与过拟合迹象。 
5. **检索与评测**

   * 先用 (\Omega) 对库做**Top-N召回**（点积相似度）；再用展平且排序后的 (\Phi) 做**余弦相似度**排序；统计 **R@1/R@5**。 

---

## 复现时的坑与注意

* **数据分布偏差**：University-1652 的UAV图来自GE 3D合成，存在**视角围绕中心旋转**的偏置，导致特征/节点位置可能集中在图像中心——注意正负样本挖掘和泛化评估（可用SenseFly零样本测试）。 
* **结构位置信息的语义性**：节点坐标仅表示**语义结构中心**，并非可作为几何定位的精确点位。
* **维度与存储**：半全局描述子维度随 **K** 线性增长（`K×256`）；在大图库下注意内存与检索效率的权衡。

---

## 与经典 NetVLAD 的差异（复现思维导图）

* NetVLAD：残差累加 → **全局向量**，**空间信息丢失**；
* SGM-Net/CSMG：**相似度掩膜**→**聚合共识特征**→**恢复空间分布并排序**→**结构感知**的半全局向量，与**全局向量**配合层级检索。 

---

## 最小复现伪代码（训练主循环轮廓）

```python
# 架构：Backbone(Conv1-4, frozen) -> NonLocal(+conv adapt + maxpool)
#  -> Global branch (flatten) = Omega
#  -> CSMG branch = Phi -> node_area_sort_then_flatten = f

for batch in loader:  # (sat_anchors, uav_pos, uav_neg)
    x = backbone(images)                 # BxCxHxW
    xg = nonlocal_block(x)               # Bx256x14x14
    Omega = xg.flatten(1)                # Bx(256*14*14)

    # CSMG
    S = bmm(C.transpose(0,1), X)         # ~ ReLU(c^T x) with proper shaping
    S = relu(S)
    Xstar = S.unsqueeze(1) * X_expanded  # elementwise mask
    d = Xstar.sum(dim=-1)                # BxDxK
    Phi = sort_nodes_by_area_and_flatten(d, S)  # -> Bx(K*256) = f

    # Loss
    L_cos = cosine_triplet(f_anchor, f_pos, f_neg, margin=alpha)
    L_glb = cosine_embedding_loss(Omega_anchor, Omega_pos/neg, ...)
    L = 0.5 * L_glb + 0.5 * L_cos

    L.backward(); optimizer.step()
```

> 上述步骤对应论文中的 **Eq.(4)(5)(6)**、节点排序与展平、余弦三元组与总损失（Eq.(9)(10)）。实现时请严格按上文“训练食谱”设置超参、数据与采样。  

---

## 推理与部署（与仓库的推理代码对齐）

* **离线**用训练好的网络对**参考地图**编码（(\Omega^r,\Phi^r)）并缓存；在线对**UAV查询**取 ((\Omega^q,\Phi^q))，先**全局Top-N**，再**半全局重排**。可直接参考论文给出的 **Algorithm 1**。
* 若需边缘部署与高吞吐：优先 **K=4**，VGG-16 backbone；该配置在 3090/NX 上性能已在文中给出。 

---

**以上内容聚焦于训练与评测所需的一手细节，便于在 `scene-graph-matching-demo` 的基础上补齐训练脚本与流水线。**
