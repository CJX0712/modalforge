# ModalForge · 多模态跨模态检索系统

> 世界顶级多模态 AI 系统：纯 numpy 离线兜底引擎 + scikit-learn CCA/PLS 顶级开源后端 + 可选 CLIP SOTA 后端，旗舰 **MMFuse** 置信门控跨模态融合检索。
> **作者：晨星** · License：MIT · Python 3.10+

---

## 一句话定位

把「图像」和「文本」投影到**同一个共享语义空间**，使「图→文」与「文→图」双向检索皆可度量。系统随机复用了多模态领域的顶级方法（典型相关分析 CCA / 偏最小二乘 PLS / 对比学习 InfoNCE），并以一个原创的**置信门控非劣融合（MMFuse）**作为旗舰，把多个对齐空间的优势结合起来。

---

## 特性

- **零下载、可离线**：合成图文语料由 PIL 程序化生成（形状 / 颜色 / 数量组合渲染为小图 + 模板 caption），固定 seed 完全可复现，无需任何网络或预训练权重。
- **三级后端，优雅降级**：
  - 纯 numpy 手写 CCA（与 scikit-learn **交叉验证**，典型相关误差 < 1e-4）
  - 纯 numpy 双编码器对比学习（CLIP 风格 InfoNCE，闭式梯度）
  - scikit-learn `CCA` / `PLSCanonical`（顶级开源，CPU 友好）
  - 可选 `transformers` CLIP（现代 SOTA，缺依赖时自动跳过）
- **双向检索评测**：recall@k、mAP、MRR，图文两个方向对称。
- **旗舰 MMFuse**：多对齐空间置信门控互逆排序融合（RRF），并带**非劣兜底**（无成员自信时退回最强单对齐器）。
- **模块化、契约优先**：`cli → pipeline → {data, encoders, alignment, retrieval, fusion} → core`，单向无环。

---

## 快速开始

```bash
# 1. 建隔离环境（推荐）
python -m venv .venv && .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. 跑端到端演示（生成语料 → 训练对齐器 → 跨模型 benchmark → 落盘 benchmark.json）
python -m modalforge.examples.run_demo

# 3. 或走 CLI
python -m modalforge.cli --seed 20261002 --out benchmark.json
```

无任何重型依赖即可运行：numpy / scipy / scikit-learn / pillow / pytest。

---

## 架构总览

```
                ┌─────────────────────────────────────────────┐
  合成图文语料  │  data.synthetic  (PIL 渲染 + 模板 caption)    │
  (离线可复现)  └─────────────────────────────────────────────┘
                          │  images + captions
            ┌─────────────┴─────────────┐
            ▼                            ▼
   encoders.HogImageEncoder     encoders.TfidfEncoder
   (HOG 形状 + 逐格 RGB 颜色)    (TF-IDF 文本)
            └─────────────┬─────────────┘
                          ▼  X_img, X_txt
        ┌──────────────────────────────────────────┐
        │  alignment (每个对齐器独立 fit/transform)  │
        │   • NumpyCCA  (手写, 与 sklearn 互验)      │
        │   • SklearnCCA / SklearnPLS               │
        │   • NumpyContrastive (InfoNCE 双编码器)   │
        │   • ClipAligner (可选 SOTA)               │
        └──────────────────────────────────────────┘
                          │  (Z_img, Z_txt) 共享空间
            ┌─────────────┴─────────────┐
            ▼                            ▼
   retrieval (cosine + recall@k/      fusion.MMFuse
   mAP/MRR, 双向)                      (置信门控 RRF 融合)
            └─────────────┬─────────────┘
                          ▼
              pipeline.benchmark() → benchmark.json
```

详见 [`docs/architecture.md`](docs/architecture.md)。

---

## 关键技术选型（随机抽选）

| 角色 | 选型 | 说明 |
|------|------|------|
| 图像编码器 | 手写 HOG + 逐格均值 RGB | 同时捕捉**形状**与**颜色**，纯 numpy |
| 文本编码器 | 手写 TF-IDF | 纯 numpy，稳定可复现 |
| 对齐后端（经典 SOTA） | scikit-learn `CCA` / `PLSCanonical` | 典型相关分析，多模态对齐基石 |
| 对齐后端（现代 SOTA） | 对比学习 InfoNCE（手写） | CLIP 风格双编码器，闭式梯度 |
| 对齐后端（可选顶级） | `transformers` CLIP | 真实图文联合表征，缺依赖自动降级 |
| 旗舰融合 | **MMFuse** 置信门控 RRF | 跨空间非劣融合 + 兜底 |

---

## 验收指标（示例，固定 seed 可复现）

运行 `python -m modalforge.examples.run_demo` 后，`benchmark.json` 内含完整跨对齐器对比表。MMFuse 的 `recall@1`（图→文 / 文→图）应**显著高于** `random_baseline`（随机检索上限）。

---

## 一键复现

```bash
make venv      # 建隔离环境并装依赖
make test      # 跑 pytest 单测（含 CCA 与 sklearn 交叉验证不变量）
make demo      # 端到端演示并生成 benchmark.json
make bench     # 仅生成 benchmark.json
```

---

## 作者与许可

- **作者**：晨星（GitHub: [CJX0712](https://github.com/CJX0712)）
- **License**：MIT
