# ✦ 晨星AI · 核心内核 v1

> 世界顶级数学内核 —— 零依赖 · 可运行 · 可复现 · 可验证
> **作者：晨星** · 许可：MIT

一个**单文件、零依赖、零构建**的 AI 数学内核。同一份 `chenxing-ai.js` 既能在 Node 里跑，也能在浏览器里跑（`window.CXAI`）。它不是调包封装，而是**手写实现世界顶级算法**，并用 11 条「可验证不变量」交叉验证，全部通过。

---

## 设计哲学（零重写 / 集大成）

不臆造轮子。每一行都站在世界顶级成果之上，并以不变量验证：

| 能力 | 世界顶级来源 | 本内核做法 | 不变量 |
|---|---|---|---|
| SVD | Jacobi 旋转 | 对 `AᵀA` 做 Jacobi 特征分解得右奇异向量 | 重构误差 < 1e-9；`UᵀU=I`、`VᵀV=I` |
| Cholesky | 经典 LLᵀ | 半正定分解 | `LLᵀ=A` 误差 < 1e-9 |
| logsumexp | 防溢出事实标准 | `max + log Σ exp(x−max)` | 大数场景不溢出、值精确 |
| 自动微分 | PyTorch/JAX 反向模式 | 手写张量计算图 + 拓扑反传 | 中心差分梯度检验 maxRelErr < 1e-5 |
| Adam | Kingma & Ba, 2015 | 精确更新 + 梯度裁剪 | 训练损失单调下降 > 10× |
| 注意力 | Vaswani et al., 2017 | 缩放点积 + softmax | 梯度检验通过 |
| Transformer | 同上 | 位置编码 + 前馈 + 残差 + LayerNorm | 反转任务 acc > 0.9 |
| PCA | Eckart–Young 最优低秩 | 协方差特征分解 | 前 k 主成分方差解释率 > 0.99 |
| 聚类/近邻 | Lloyd / Cover–Hart | k-means / k-NN | 惯性单调不增 |

**核心主张**：每个算法至少有一条能被独立实现交叉验证的硬金标准。解析解 vs 数值解、参照实现 vs 主实现、梯度检验、对数似然单调、重构误差有界——全绿才交付。

---

## 快速开始

```bash
# 复现 11/11 不变量（约 6 秒）
node chenxing-ai.js

# 或
npm test

# 重新生成单文件浏览器演示 index.html
node tools/build.js
```

**浏览器**：直接用浏览器打开 `index.html`（单文件、内联全部 CSS/JS、零外部依赖）。点击「运行自检套件」即在浏览器内实时跑完 11 条不变量，并附逻辑回归决策边界实时演示。

---

## 自检套件（11/11 ✅）

```
================================================================
  晨星AI · 核心内核 v1  ——  自检套件
================================================================
  [PASS] SVD 重构误差 < 1e-9                    (reconErr=7.77e-16)
  [PASS] SVD 正交性 ∥UᵀU−I∥<1e-9               (ortho=1.06e-11)
  [PASS] Cholesky 重构误差 < 1e-9               (err=8.88e-16)
  [PASS] logsumexp 防溢出 (≈1001.10)            (naive=Infinity)
  [PASS] LayerNorm 行均值≈0                     (maxMean=3.21e-17)
  [PASS] LayerNorm 行方差≈1                     (maxVar=1.55e-5)
  [PASS] 注意力梯度检验 maxRelErr<1e-5          (maxRel=1.55e-9)
  [PASS] 逻辑回归梯度检验 maxRelErr<1e-5        (maxRel=3.67e-12)
  [PASS] 逻辑回归损失下降 >10×                  (init=-0.674 final=-0.050)
  [PASS] PCA 前2主成分解释率>0.99               (capVar=0.9999)
  [PASS] Transformer 反转任务 acc>0.9           (acc=1.000 loss=-0.000)
================================================================
  结果: 11/11 通过  ✅ 全绿
================================================================
```

> Transformer 用的是**注意力反转任务**：位置 i 必须通过注意力从位置 `L−1−i` 取回 token。这是需要真正学会「内容寻址」的任务（而非可被前馈层糊弄的恒等复制），模型收敛到 acc=1.000。

---

## 架构一览

```
chenxing-ai.js  (单文件)
├── 线性代数        matmul / transpose / SVD(Jacobi) / Cholesky / eye
├── 数值稳定        logsumexp
├── 自动微分 T      计算图节点：matmul/add/sub/mul/scale/transpose/
│                   relu/softmax/softmaxMasked/layernorm/log/row/sum
│                   + backward() 拓扑反传 + zeroGrads/clipGrads
├── 优化器 Adam      Kingma&Ba 更新 + 梯度裁剪(clipGrads)
├── 经典 ML         pca / kmeans / knnPredict / logisticTrain
├── 注意力           attention(x,Wq,Wk,Wv,dk)  缩放点积
├── Transformer     demoTransformer()  位置编码+注意力+FFN+残差+LN
└── 自检套件         runSelfTests()  11 条不变量, 退出码非 0 表示失败
```

---

## 作为库使用

```js
const CX = require('./chenxing-ai.js');   // Node
// 或浏览器: const CX = window.CXAI;

// 1) 数值稳定的对数和指数
CX.logsumexp([1000, 1000.1, 999.9]);      // 1001.1019（naive 会溢出成 Infinity）

// 2) SVD
const { U, S, V } = CX.svd(A);            // A ≈ U·diag(S)·Vᵀ

// 3) 自动微分 + 梯度检验
const x = new CX.T([[1,2],[3,4]]);
const w = new CX.T(CX.randnMat(2,2,0.1));
const loss = CX.T.sum(CX.T.mul(CX.T.matmul(x,w), CX.T.matmul(x,w)));
CX.zeroGrads([w]); CX.backward(loss);     // w.grad 即解析梯度

// 4) PCA 降维
const r = CX.pca(X, 2);                   // r.proj (N×2), r.capVar 方差解释率

// 5) 训练一个会「反转序列」的 Transformer
const res = CX.demoTransformer();         // { loss, acc }  acc→1.000
```

---

## 目录结构

```
chenxing-ai-core/
├── chenxing-ai.js     # 内核 + 自检套件（单文件，零依赖）
├── index.html         # 单文件浏览器演示（内联全部 CSS/JS）
├── tools/build.js     # 把内核原样内联进 index.html（可复现构建）
├── tools/template.html# index.html 模板
├── package.json
├── LICENSE            # MIT
└── README.md
```

---

## 复现性与验证

- **确定性**：内置 mulberry32 种子随机，`runSelfTests()` 以固定种子运行，结果逐位可复现。
- **可验证**：每条不变量都可被独立实现交叉验证（见上表）。`node chenxing-ai.js` 全绿即通过，失败则进程以非 0 退出（便于 CI）。
- **双环境**：同一引擎在 Node 与浏览器一致运行；`index.html` 由 `tools/build.js` 从 `chenxing-ai.js` 生成，杜绝双份代码漂移。

---

## 作者

**晨星** —— 启明星，天亮前最后还亮着的那颗。夜里把活干完，天亮交出真能用的东西。

✨ MIT · 2026
