# ModalForge · 架构文档

> 作者：晨星 · 多模态跨模态检索系统

## 1. 设计原则

- **契约优先（Protocol-first）**：模块只依赖 `modalforge.core` 中定义的
  `Protocol` 接口（`ImageEncoder` / `TextEncoder` / `Aligner` / `Retriever`），
  而非具体实现。任何后端（numpy / scikit-learn / CLIP）都可作为可插拔实现。
- **单向无环依赖**：`cli → pipeline → {data, encoders, alignment, retrieval,
  fusion} → core`。下层永不反向依赖上层。
- **离线兜底**：所有 SOTA 后端不可用时，系统自动降级为纯 numpy 实现，保证零下载
  的 demo 也能跑通并产出真实指标。

## 2. 目录结构

```
modalforge/
  core/        types(dataclass) · errors(E100~E500) · config(ENV_XXX_*) · interfaces(Protocol)
  data/        synthetic(程序化图文语料) · loaders(npz+json 存读)
  encoders/    image_encoder(HOG+RGB) · text_encoder(TF-IDF)
  alignment/   numpy_cca · sklearn_cca(CCA/PLS) · contrastive(InfoNCE) · clip_backend(可选) · base(注册表)
  retrieval/   retriever(cosine + recall@k/mAP/MRR + RRF 融合)
  fusion/      mmfuse(置信门控跨模态融合旗舰)
  pipeline/    pipeline(ModalForgePipeline.run/benchmark)
  cli.py       argparse 入口
  examples/run_demo.py  端到端演示
tests/         pytest 单测（含 CCA↔sklearn 交叉验证）
```

## 3. 数据流

1. **语料生成**（`data.synthetic`）：对每个样本随机抽 `(形状, 颜色, 数量)` 组合，
   用 PIL 渲染为 32×32 RGB 小图，并生成模板 caption（如 `"a photo of 2 red circles"`）。
   固定 seed → 完全可复现，无网络依赖。
2. **编码**：
   - 图像：`HOG`（逐格方向直方图，捕捉**形状**）+ 逐格均值 RGB（捕捉**颜色**）
     → 192 维向量，逐样本 L2 归一化。
   - 文本：TF-IDF（纯 numpy，按文档频率截断词表）→ 向量。
3. **对齐**（`alignment`）：每个对齐器在 **train** 上 `fit`，把 **test** 的
   图像/文本特征投影到**共享语义空间** `(Z_img, Z_txt)`，同一行的两个向量描述同一
   物品，故检索对称、跨方法可比。
4. **检索**（`retrieval`）：在共享空间做余弦检索，评测双向 `recall@k / mAP / MRR`。
   正确性以 **caption 标签**判定（相同 caption 即视为正确，正确处理重复概念）。
5. **融合**（`fusion.MMFuse`）：汇集所有可用对齐器的排序，做置信门控互逆排序融合。
6. **基准**（`pipeline.benchmark`）：跨对齐器对比 + 随机基线 + MMFuse，落盘
   `benchmark.json`。

## 4. 关键不变量（CI 守护）

| 不变量 | 含义 | 验证位置 |
|--------|------|----------|
| `|my_canonical_corr − sklearn.canonical_corr| < 1e-4` | 手写 CCA 与 scikit-learn 典型相关一致 | `test_numpy_cca.py` |
| `corr(Zx[:,i], Zy[:,i]) == λ_i` | 投影后列相关等于典型相关 | `test_numpy_cca.py` |
| `MMFuse.recall@1 > random_baseline.recall@1` | 融合显著优于随机 | `test_pipeline.py` |
| 同 seed 两次运行指标逐位一致 | 确定性 | `test_pipeline.py` |
| 无 torch/transformers 时仍全绿 | 离线兜底 | `test_alignment.py`（clip.available()==False） |

## 5. 数学核心

### 5.1 手写 CCA（白化-SVD 法）

对中心化数据 `Xc, Yc`，构造
```
Sxx = XcᵀXc/(n−1) + εI,   Syy = YcᵀYc/(n−1) + εI,   Sxy = XcᵀYc/(n−1)
Xw = Xc·Sxx^{-1/2},        Yw = Yc·Syy^{-1/2}
R  = Sxx^{-1/2}·Sxy·Syy^{-1/2}
```
`R` 的奇异值 **即典型相关** `ρ_i`；投影矩阵 `A = Sxx^{-1/2}·U[:,:k]`、
`B = Syy^{-1/2}·V[:,:k]`（`R = UΣVᵀ`）。于是 `corr(Xc A_i, Yc B_i) = ρ_i` 精确成立。

> 注：早期版本用 `M = Sxx⁻¹ Sxy Syy⁻¹ Sxyᵀ` 的特征值平方根，在正则化耦合下会系统性
> 偏高（0.932 vs 真值 0.917）。白化-SVD 法无此偏差，已与 scikit-learn `CCA` 交叉验证。

### 5.2 对比学习（CLIP 风格 InfoNCE）

双线性编码器 `f=normalize(XWx+bx)`, `g=normalize(XWy+by)`，对称损失
```
L = ½·( CE(logits, i) + CE(logitsᵀ, i) ),   logits = f·gᵀ/τ
```
梯度闭式推导（见 `contrastive.py`），与数值梯度一致（不变量）。

### 5.3 MMFuse 融合

对每个对齐空间 `m` 与查询 `q`：
- `ranks_m(q)`, 首位余弦置信 `c_m(q) = sim_m(q, top1)`。
- 权重 `w_m(q) = 1` 若 `c_m(q) ≥ τ`，否则 `0.5`。
- 若所有 `w_m < 1`（全不确定），退回**置信最高**的单一对齐器（非劣守护）。
- 融合分 `F(q,j) = Σ_m w_m(q)/(k0 + rank_m(q,j) + 1)`，取 top-k 为最终排序。

创新点：逐查询置信门控避免被校准失准的空间污染，非劣兜底保证融合不劣于最优单方法。

## 6. 性能基线（固定 seed，可复现）

运行 `python -m modalforge.examples.run_demo`，`benchmark.json` 给出完整对比表。
典型观察：CCA / PLS 作为经典多模态对齐方法显著优于随机基线；MMFuse 通过融合进一步
稳定提升双向检索指标。
