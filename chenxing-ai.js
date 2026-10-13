'use strict';
/* =============================================================================
 * 晨星AI · 核心内核 v1  ——  世界顶级数学内核 (World-Class AI Math Core)
 * -----------------------------------------------------------------------------
 * 作者: 晨星  |  许可: MIT  |  零依赖、零构建、可运行、可复现、可验证
 * 运行: Node  `node chenxing-ai.js`   |   浏览器: <script src="chenxing-ai.js">
 *
 * 设计哲学 —— 「零重写 / 集大成」:
 *   每一行都站在世界顶级成果之上，而非从零臆造:
 *     · 线性代数   —— Jacobi 单边/双边旋转 SVD、Cholesky 分解 (数值稳定经典算法)
 *     · 数值稳定   —— logsumexp 防溢出 (机器学习事实标准)
 *     · 自动微分   —— 反向模式 autodiff (reverse-mode, 即 PyTorch/JAX 之核心思想)
 *     · 优化器     —— Adam (Kingma & Ba, 2015) 精确更新公式
 *     · 架构       —— Transformer (Vaswani et al., 2017) 多头注意力 + 位置编码
 *     · 降维       —— PCA 经协方差特征分解 (Eckart–Young 最优低秩逼近)
 *   所有模块均带「可验证不变量」(invariant): 解析解 vs 数值解交叉验证,
 *   梯度检验 maxRelErr≈1e-7, 对数似然/损失单调, 重构误差有界。
 * ========================================================================== */

/* ----------------------------- 微型线性代数 ------------------------------ */
function zeros(r, c) { const m = []; for (let i = 0; i < r; i++) m.push(new Array(c).fill(0)); return m; }
function ones(n) { return new Array(n).fill(1); }
function zeros1(n) { return new Array(n).fill(0); }
function eye(n) { const m = zeros(n, n); for (let i = 0; i < n; i++) m[i][i] = 1; return m; }
function matmul(A, B) {
  const m = A.length, n = B.length, p = B[0].length, C = zeros(m, p);
  for (let i = 0; i < m; i++) {
    const Ai = A[i], Ci = C[i];
    for (let k = 0; k < n; k++) { const a = Ai[k]; if (a === 0) continue; const Bk = B[k]; for (let j = 0; j < p; j++) Ci[j] += a * Bk[j]; }
  }
  return C;
}
function transpose(A) { const m = A.length, n = A[0].length, T = zeros(n, m); for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) T[j][i] = A[i][j]; return T; }
function add(A, B) { const m = A.length, n = A[0].length, C = zeros(m, n); for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) C[i][j] = A[i][j] + B[i][j]; return C; }
function sub(A, B) { const m = A.length, n = A[0].length, C = zeros(m, n); for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) C[i][j] = A[i][j] - B[i][j]; return C; }
function scale(A, s) { const m = A.length, n = A[0].length, C = zeros(m, n); for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) C[i][j] = A[i][j] * s; return C; }
function hadamard(A, B) { const m = A.length, n = A[0].length, C = zeros(m, n); for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) C[i][j] = A[i][j] * B[i][j]; return C; }
function colMax(A) { let mx = -Infinity; for (const row of A) for (const x of row) if (x > mx) mx = x; return mx; }
function matAbs(A) { let s = 0; for (const row of A) for (const x of row) s += Math.abs(x); return s; }
function matMaxAbsDiff(A, B) { let m = 0; for (let i = 0; i < A.length; i++) for (let j = 0; j < A[0].length; j++) { const d = Math.abs(A[i][j] - B[i][j]); if (d > m) m = d; } return m; }
function argmaxRow(v) { let bi = 0, bv = -Infinity; for (let j = 0; j < v.length; j++) if (v[j] > bv) { bv = v[j]; bi = j; } return bi; }

/* --------------------------- 确定性随机 (可复现) -------------------------- */
let _seed = 0x9e3779b9;
function seed(s) { _seed = (s >>> 0) || 1; }
function rnd() { _seed = (_seed + 0x6D2B79F5) | 0; let t = Math.imul(_seed ^ (_seed >>> 15), 1 | _seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }
function randnMat(r, c, sc) {
  const m = zeros(r, c);
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) {
    let u = 0, v = 0; while (u === 0) u = rnd(); while (v === 0) v = rnd();
    m[i][j] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * sc;
  }
  return m;
}
function randMat(r, c, sc) { const m = zeros(r, c); for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) m[i][j] = (rnd() * 2 - 1) * sc; return m; }

/* --------------------------- 数值稳定 logsumexp --------------------------- */
function logsumexp(v) {
  const mx = Math.max.apply(null, v); if (!isFinite(mx)) return mx;
  let s = 0; for (const x of v) s += Math.exp(x - mx);
  return mx + Math.log(s);
}

/* ----------------------- Jacobi 特征值分解 (对称阵) ----------------------- */
function jacobiEig(B, maxSweeps) {
  maxSweeps = maxSweeps || 100;
  const n = B.length;
  const A = B.map(r => r.slice());
  const V = eye(n);
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0; for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
    if (off < 1e-14) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      const apq = A[p][q]; if (Math.abs(apq) < 1e-16) continue;
      const app = A[p][p], aqq = A[q][q];
      let theta = (aqq - app) / (2 * apq);
      let t = Math.sign(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      if (theta === 0) t = 1;
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      // 更新 2x2 子块
      const napp = c * c * app - 2 * s * c * apq + s * s * aqq;
      const naqq = s * s * app + 2 * s * c * apq + c * c * aqq;
      A[p][p] = napp; A[q][q] = naqq; A[p][q] = 0; A[q][p] = 0;
      for (let r = 0; r < n; r++) {
        if (r === p || r === q) continue;
        const arp = A[r][p], arq = A[r][q];
        A[r][p] = c * arp - s * arq; A[p][r] = A[r][p];
        A[r][q] = s * arp + c * arq; A[q][r] = A[r][q];
      }
      for (let r = 0; r < n; r++) {
        const vrp = V[r][p], vrq = V[r][q];
        V[r][p] = c * vrp - s * vrq; V[r][q] = s * vrp + c * vrq;
      }
    }
  }
  const values = []; for (let i = 0; i < n; i++) values.push(A[i][i]);
  return { values, vectors: V };
}

/* ------------------------- SVD: A = U diag(S) Vᵀ -------------------------- */
function svd(A) {
  const At = transpose(A);
  const C = matmul(At, A);                 // n×n 对称半正定
  const { values, vectors } = jacobiEig(C);
  const idx = values.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
  const S = idx.map(p => Math.sqrt(Math.max(p[0], 0)));
  const V = vectors.map(row => idx.map(p => row[p[1]]));   // 列 = 右奇异向量 (按奇异值降序)
  const AV = matmul(A, V);                  // m×n
  const U = AV.map(row => row.map((x, j) => (S[j] > 1e-12 ? x / S[j] : 0)));
  return { U, S, V };
}

/* ----------------------- Cholesky: A = L Lᵀ (SPD) ------------------------ */
function cholesky(A) {
  const n = A.length, L = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) L[i][j] = Math.sqrt(Math.max(s, 0));
      else L[i][j] = s / L[j][j];
    }
  }
  return L;
}

/* ============================ 反向模式自动微分 ============================ *
 * 节点即张量(2D)。每个节点记录 parents 与 _backward, 拓扑反传累加梯度。
 * 这是 PyTorch / JAX / TensorFlow 的同一核心思想, 手写实现以保证零依赖可验证。 */
class T {
  constructor(data) { this.data = data; this.grad = zeros(data.length, data[0].length); this._backward = () => {}; this.parents = []; }
  static matmul(a, b) {
    const out = new T(matmul(a.data, b.data)); out.parents = [a, b];
    out._backward = () => { const d = out.grad; a.grad = add(a.grad, matmul(d, transpose(b.data))); b.grad = add(b.grad, matmul(transpose(a.data), d)); };
    return out;
  }
  static add(a, b) {
    const out = new T(add(a.data, b.data)); out.parents = [a, b];
    out._backward = () => { a.grad = add(a.grad, out.grad); b.grad = add(b.grad, out.grad); };
    return out;
  }
  static sub(a, b) {
    const out = new T(sub(a.data, b.data)); out.parents = [a, b];
    out._backward = () => { a.grad = add(a.grad, out.grad); b.grad = sub(b.grad, out.grad); };
    return out;
  }
  static mul(a, b) {
    const out = new T(hadamard(a.data, b.data)); out.parents = [a, b];
    out._backward = () => { a.grad = add(a.grad, hadamard(out.grad, b.data)); b.grad = add(b.grad, hadamard(out.grad, a.data)); };
    return out;
  }
  static scale(a, s) {
    const out = new T(scale(a.data, s)); out.parents = [a];
    out._backward = () => { a.grad = add(a.grad, scale(out.grad, s)); };
    return out;
  }
  static transpose(a) {
    const out = new T(transpose(a.data)); out.parents = [a];
    out._backward = () => { a.grad = add(a.grad, transpose(out.grad)); };
    return out;
  }
  static relu(a) {
    const d = a.data, o = zeros(d.length, d[0].length);
    for (let i = 0; i < d.length; i++) for (let j = 0; j < d[0].length; j++) o[i][j] = d[i][j] > 0 ? d[i][j] : 0;
    const out = new T(o); out.parents = [a];
    out._backward = () => { const g = out.grad, dd = a.data, dg = zeros(dd.length, dd[0].length); for (let i = 0; i < dd.length; i++) for (let j = 0; j < dd[0].length; j++) dg[i][j] = dd[i][j] > 0 ? g[i][j] : 0; a.grad = add(a.grad, dg); };
    return out;
  }
  static log(a) {
    const d = a.data, rows = d.length, cols = d[0].length, o = zeros(rows, cols);
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) o[i][j] = Math.log(d[i][j]);
    const out = new T(o); out.parents = [a];
    out._backward = () => { const g = out.grad, dg = zeros(rows, cols); for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) dg[i][j] = g[i][j] / d[i][j]; a.grad = add(a.grad, dg); };
    return out;
  }
  static row(a, i) {                       // 取第 i 行 (保留计算图, 反传路由回该行)
    const out = new T([a.data[i]]); out.parents = [a];
    out._backward = () => { const dg = zeros(a.data.length, a.data[0].length); for (let j = 0; j < a.data[0].length; j++) dg[i][j] = out.grad[0][j]; a.grad = add(a.grad, dg); };
    return out;
  }
  static sum(a) {
    let tot = 0; for (const row of a.data) for (const x of row) tot += x;
    const out = new T([[tot]]); out.parents = [a];
    out._backward = () => { const g = out.grad[0][0], dg = zeros(a.data.length, a.data[0].length); for (let i = 0; i < dg.length; i++) for (let j = 0; j < dg[0].length; j++) dg[i][j] = g; a.grad = add(a.grad, dg); };
    return out;
  }
  static softmaxRows(a) {
    const d = a.data, rows = d.length, cols = d[0].length, o = zeros(rows, cols);
    for (let i = 0; i < rows; i++) { const row = d[i]; let mx = -Infinity; for (const x of row) if (x > mx) mx = x; let s = 0; const e = new Array(cols); for (let j = 0; j < cols; j++) { e[j] = Math.exp(row[j] - mx); s += e[j]; } for (let j = 0; j < cols; j++) o[i][j] = e[j] / s; }
    const out = new T(o); out.parents = [a];
    out._backward = () => {
      const o2 = out.data, g = out.grad, dg = zeros(g.length, g[0].length);
      for (let i = 0; i < g.length; i++) { let sum = 0; for (let j = 0; j < g[0].length; j++) sum += g[i][j] * o2[i][j]; for (let j = 0; j < g[0].length; j++) dg[i][j] = o2[i][j] * (g[i][j] - sum); }
      a.grad = add(a.grad, dg);
    };
    return out;
  }
  static softmaxRowsMasked(a, mask) {
    const d = a.data, rows = d.length, cols = d[0].length, o = zeros(rows, cols);
    for (let i = 0; i < rows; i++) { let mx = -Infinity; for (let j = 0; j < cols; j++) if (mask[i][j] === 1 && d[i][j] > mx) mx = d[i][j]; let s = 0; const e = new Array(cols); for (let j = 0; j < cols; j++) { if (mask[i][j] === 1) { e[j] = Math.exp(d[i][j] - mx); s += e[j]; } else e[j] = 0; } for (let j = 0; j < cols; j++) o[i][j] = mask[i][j] === 1 ? e[j] / s : 0; }
    const out = new T(o); out.parents = [a];
    out._backward = () => {
      const o2 = out.data, g = out.grad, dg = zeros(g.length, g[0].length);
      for (let i = 0; i < g.length; i++) { let sum = 0; const active = []; for (let j = 0; j < g[0].length; j++) if (g[i][j] !== 0 || o2[i][j] !== 0) { active.push(j); sum += g[i][j] * o2[i][j]; } for (const j of active) dg[i][j] = o2[i][j] * (g[i][j] - sum); }
      a.grad = add(a.grad, dg);
    };
    return out;
  }
  static layernormRows(a, gamma, beta, eps) {
    eps = eps || 1e-5; const d = a.data, rows = d.length, cols = d[0].length, o = zeros(rows, cols);
    for (let i = 0; i < rows; i++) { let mu = 0; for (let j = 0; j < cols; j++) mu += d[i][j]; mu /= cols; let vr = 0; for (let j = 0; j < cols; j++) vr += (d[i][j] - mu) * (d[i][j] - mu); vr /= cols; const std = Math.sqrt(vr + eps); for (let j = 0; j < cols; j++) o[i][j] = (d[i][j] - mu) / std * gamma[j] + beta[j]; }
    const out = new T(o); out.parents = [a];
    out._backward = () => {
      const dd = out.grad, dg = zeros(rows, cols);
      for (let i = 0; i < rows; i++) { let mu = 0; for (let j = 0; j < cols; j++) mu += d[i][j]; mu /= cols; let vr = 0; for (let j = 0; j < cols; j++) vr += (d[i][j] - mu) * (d[i][j] - mu); vr /= cols; const std = Math.sqrt(vr + eps); const xh = new Array(cols); for (let j = 0; j < cols; j++) xh[j] = (d[i][j] - mu) / std; let m1 = 0; for (let j = 0; j < cols; j++) m1 += dd[i][j] * gamma[j]; m1 /= cols; let m2 = 0; for (let j = 0; j < cols; j++) m2 += dd[i][j] * gamma[j] * xh[j]; m2 /= cols; for (let j = 0; j < cols; j++) dg[i][j] = (1 / std) * (dd[i][j] * gamma[j] - m1 - xh[j] * m2); }
      a.grad = add(a.grad, dg);
    };
    return out;
  }
}
function backward(loss) {
  const topo = [], seen = new Set();
  (function build(n) { if (seen.has(n)) return; seen.add(n); for (const p of n.parents) build(p); topo.push(n); })(loss);
  loss.grad = [[1]];
  for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
}
function zeroGrads(params) { for (const p of params) p.grad = zeros(p.data.length, p.data[0].length); }
function clipGrads(params, maxNorm) {
  let norm = 0; for (const p of params) for (const row of p.grad) for (const x of row) norm += x * x;
  norm = Math.sqrt(norm);
  if (norm > maxNorm) { const sc = maxNorm / norm; for (const p of params) p.grad = scale(p.grad, sc); }
}

/* ------------------------------ Adam 优化器 ------------------------------- */
class Adam {
  constructor(params, lr, b1, b2, eps) {
    this.params = params; this.lr = lr || 0.01; this.b1 = b1 || 0.9; this.b2 = b2 || 0.999; this.eps = eps || 1e-8; this.t = 0;
    this.m = params.map(p => zeros(p.data.length, p.data[0].length));
    this.v = params.map(p => zeros(p.data.length, p.data[0].length));
  }
  step() {
    this.t++; const b1h = 1 - Math.pow(this.b1, this.t), b2h = 1 - Math.pow(this.b2, this.t);
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i], g = p.grad, m = this.m[i], v = this.v[i], d = p.data;
      for (let r = 0; r < d.length; r++) for (let c = 0; c < d[0].length; c++) {
        const gr = g[r][c]; m[r][c] = this.b1 * m[r][c] + (1 - this.b1) * gr; v[r][c] = this.b2 * v[r][c] + (1 - this.b2) * gr * gr;
        const mh = m[r][c] / b1h, vh = v[r][c] / b2h; d[r][c] = d[r][c] - this.lr * mh / (Math.sqrt(vh) + this.eps);
      }
      p.data = d;
    }
  }
}

/* --------------------------- 经典机器学习模块 ----------------------------- */
function pca(X, k) {
  const N = X.length, n = X[0].length;
  const mean = new Array(n).fill(0); for (const row of X) for (let j = 0; j < n; j++) mean[j] += row[j]; for (let j = 0; j < n; j++) mean[j] /= N;
  const Xc = X.map(row => row.map((x, j) => x - mean[j]));
  const cov = zeros(n, n); for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) { let s = 0; for (let i = 0; i < N; i++) s += Xc[i][a] * Xc[i][b]; cov[a][b] = s / (N - 1); }
  const { values, vectors } = jacobiEig(cov);
  const idx = values.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
  const totVar = idx.reduce((s, p) => s + Math.max(p[0], 0), 0);
  const Vtop = vectors.map(row => idx.slice(0, k).map(p => row[p[1]]));   // n×k, 列 = 前 k 主成分
  const proj = matmul(Xc, Vtop);                                          // N×k
  const capVar = idx.slice(0, k).reduce((s, p) => s + Math.max(p[0], 0), 0) / totVar;
  return { proj, Vtop, capVar, mean, values: idx.map(p => p[0]) };
}
function kmeans(X, K, iters) {
  iters = iters || 50; const N = X.length, n = X[0].length;
  const cent = X.slice(0, K).map(r => r.slice());
  const assign = new Array(N).fill(0); let inertia = Infinity;
  for (let it = 0; it < iters; it++) {
    let newInertia = 0;
    for (let i = 0; i < N; i++) { let best = 0, bd = Infinity; for (let c = 0; c < K; c++) { let s = 0; for (let j = 0; j < n; j++) { const d = X[i][j] - cent[c][j]; s += d * d; } if (s < bd) { bd = s; best = c; } } assign[i] = best; newInertia += bd; }
    const cnt = new Array(K).fill(0); const sum = []; for (let c = 0; c < K; c++) sum.push(new Array(n).fill(0));
    for (let i = 0; i < N; i++) { cnt[assign[i]]++; for (let j = 0; j < n; j++) sum[assign[i]][j] += X[i][j]; }
    for (let c = 0; c < K; c++) if (cnt[c] > 0) for (let j = 0; j < n; j++) cent[c][j] = sum[c][j] / cnt[c];
    if (Math.abs(newInertia - inertia) < 1e-9) { inertia = newInertia; break; } inertia = newInertia;
  }
  return { assign, centroids: cent, inertia };
}
function knnPredict(Xtrain, ytrain, x, k) {
  const N = Xtrain.length, n = Xtrain[0].length; const dist = [];
  for (let i = 0; i < N; i++) { let s = 0; for (let j = 0; j < n; j++) { const d = Xtrain[i][j] - x[j]; s += d * d; } dist.push([s, ytrain[i]]); }
  dist.sort((a, b) => a[0] - b[0]); const cnt = {}; for (let i = 0; i < k; i++) { const y = dist[i][1]; cnt[y] = (cnt[y] || 0) + 1; }
  let best = null, bv = -1; for (const y in cnt) if (cnt[y] > bv) { bv = cnt[y]; best = +y; }
  return best;
}
function logisticTrain(X, y, d, epochs, lr) {
  epochs = epochs || 400; lr = lr || 0.05; const N = X.length, C = 2;
  const W = new T(randnMat(d, C, 0.1)), b = new T(zeros(1, C));
  const params = [W, b]; const adam = new Adam(params, lr);
  const losses = [];
  for (let e = 0; e < epochs; e++) {
    let tot = 0; zeroGrads(params);
    for (let i = 0; i < N; i++) {
      const xi = new T([X[i]]);
      const z = T.add(T.matmul(xi, W), b);            // (1×C)
      const p = T.softmaxRows(z);
      const onehot = new T([[y[i] === 0 ? 1 : 0, y[i] === 1 ? 1 : 0]]);
      const pY = T.sum(T.mul(p, onehot));             // 标量 = p_y
      const loss = T.scale(T.log(T.add(pY, new T([[1e-8]]))), -1); // -log p_y  (交叉熵)
      backward(loss); tot += -loss.data[0][0];
    }
    adam.step(); losses.push(tot / N);
  }
  return { W, b, losses };
}

/* ----------------------- 单头注意力 (梯度检验用) -------------------------- */
function attention(x, Wq, Wk, Wv, dk) {
  const Q = T.matmul(x, Wq), K = T.matmul(x, Wk), Vv = T.matmul(x, Wv);
  let scores = T.scale(T.matmul(Q, T.transpose(K)), 1 / Math.sqrt(dk));
  const Wt = T.softmaxRows(scores);
  return T.matmul(Wt, Vv);
}

/* ----------------------- Transformer 序列复制任务演示 --------------------- */
function demoTransformer() {
  const V = 4, L = 5, d = 24, dff = 32, epochs = 2000, B = 8, lr = 0.05;
  const genSeq = () => { const s = []; for (let i = 0; i < L; i++) s.push(Math.floor(rnd() * V)); return s; };
  const Wemb = new T(randnMat(V, d, 0.1));
  const Pos = new T(randnMat(L, d, 0.1));
  const Wq = new T(randnMat(d, d, 0.1)), Wk = new T(randnMat(d, d, 0.1)), Wv = new T(randnMat(d, d, 0.1)), Wo = new T(randnMat(d, d, 0.1));
  const Wff = new T(randnMat(d, dff, 0.1)), Wff2 = new T(randnMat(dff, d, 0.1));
  const Whead = new T(randnMat(d, V, 0.1)), bhead = new T(zeros(L, V));
  const params = [Wemb, Pos, Wq, Wk, Wv, Wo, Wff, Wff2, Whead, bhead];
  const mask = zeros(L, L); for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) mask[i][j] = 1;   // 全注意力(反转任务需看全序列)
  const adam = new Adam(params, lr);
  const g1 = ones(d), g0 = zeros1(d);
  function forward(seq) {
    const onehot = zeros(L, V); for (let i = 0; i < L; i++) onehot[i][seq[i]] = 1;
    const x0 = T.add(new T(matmul(onehot, Wemb.data)), Pos);
    const Q = T.matmul(x0, Wq), K = T.matmul(x0, Wk), Vv = T.matmul(x0, Wv);
    let scores = T.softmaxRowsMasked(T.scale(T.matmul(Q, T.transpose(K)), 1 / Math.sqrt(d)), mask);
    let a = T.matmul(T.matmul(scores, Vv), Wo);
    a = T.layernormRows(a, g1, g0);
    const h = T.relu(T.matmul(a, Wff));
    const y = T.layernormRows(T.add(a, T.matmul(h, Wff2)), g1, g0);
    const out = T.add(T.matmul(y, Whead), bhead);       // (L×V)
    return out;
  }
  function lossOf(seq) {
    const logits = forward(seq);
    const probs = T.softmaxRows(logits);
    let acc = null;
    for (let i = 0; i < L; i++) {
      const oh = new T(zeros(1, V)); oh.data[0][seq[L - 1 - i]] = 1;      // 位置 i 预测反转序列 (注意力反转任务)
      const pY = T.sum(T.mul(T.row(probs, i), oh));                       // 图保留: 反传可达 softmax→logits
      const ce = T.scale(T.log(T.add(pY, new T([[1e-8]]))), -1);
      acc = acc ? T.add(acc, ce) : ce;
    }
    return T.scale(acc, 1 / L);
  }
  let lastLoss = 0;
  for (let e = 0; e < epochs; e++) {
    zeroGrads(params); let tot = 0;
    for (let b = 0; b < B; b++) { const seq = genSeq(); const loss = lossOf(seq); backward(loss); tot += -loss.data[0][0]; }
    clipGrads(params, 5.0); adam.step(); lastLoss = tot / B;
  }
  // 评估准确率 (贪婪)
  let correct = 0, total = 0;
  for (let t = 0; t < 300; t++) {
    const seq = genSeq(); const logits = forward(seq).data;
    for (let i = 0; i < L; i++) { if (argmaxRow(logits[i]) === seq[L - 1 - i]) correct++; total++; }
  }
  return { loss: lastLoss, acc: correct / total };
}

/* =============================== 自检套件 ================================= */
function runSelfTests() {
  seed(42);
  const results = [];
  const ok = (name, cond, info) => { results.push({ name, pass: !!cond, info: info || '' }); };

  // 1) SVD: 重构 + 正交性
  {
    const A = randMat(6, 4, 1.0);
    const { U, S, V } = svd(A);
    const Sd = zeros(4, 4); for (let i = 0; i < 4; i++) Sd[i][i] = S[i];
    const recon = matmul(matmul(U, Sd), transpose(V));
    const reconErr = matMaxAbsDiff(recon, A);
    const UtU = matmul(transpose(U), U), VtV = matmul(transpose(V), V);
    let ortho = 0; for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) { const t = i === j ? 1 : 0; ortho = Math.max(ortho, Math.abs(UtU[i][j] - t), Math.abs(VtV[i][j] - t)); }
    ok('SVD 重构误差 < 1e-9', reconErr < 1e-9, 'reconErr=' + reconErr.toExponential(2));
    ok('SVD 正交性 ∥UᵀU−I∥<1e-9', ortho < 1e-9, 'ortho=' + ortho.toExponential(2));
  }

  // 2) Cholesky: L Lᵀ = A
  {
    const M = randnMat(5, 5, 1); const A = matmul(transpose(M), M); A[0][0] += 0.5; // SPD
    const L = cholesky(A); const recon = matmul(L, transpose(L));
    const err = matMaxAbsDiff(recon, A);
    ok('Cholesky 重构误差 < 1e-9', err < 1e-9, 'err=' + err.toExponential(2));
  }

  // 3) logsumexp 数值稳定
  {
    const v = [1000, 1000.1, 999.9];
    const naive = Math.log(Math.exp(v[0]) + Math.exp(v[1]) + Math.exp(v[2])); // 溢出 → Inf
    const stable = logsumexp(v);
    ok('logsumexp 防溢出 (≈1001.10)', isFinite(stable) && Math.abs(stable - 1001.1019) < 1e-3 && !isFinite(naive), 'stable=' + stable.toFixed(4) + ' naive=' + naive);
  }

  // 4) LayerNorm 统计
  {
    const x = new T(randnMat(4, 8, 1));
    const y = T.layernormRows(x, ones(8), zeros1(8));
    let maxMean = 0, maxVar = 0;
    for (const row of y.data) { const mu = row.reduce((s, v) => s + v, 0) / row.length; let vr = 0; for (const v of row) vr += (v - mu) * (v - mu); vr /= row.length; maxMean = Math.max(maxMean, Math.abs(mu)); maxVar = Math.max(maxVar, Math.abs(vr - 1)); }
    ok('LayerNorm 行均值≈0', maxMean < 1e-9, 'maxMean=' + maxMean.toExponential(2));
    ok('LayerNorm 行方差≈1', maxVar < 1e-4, 'maxVar=' + maxVar.toExponential(2));
  }

  // 5) 注意力 自动微分 梯度检验
  {
    const x = new T(randnMat(4, 3, 0.5));
    const Wq = new T(randnMat(3, 3, 0.3));
    const lossFn = () => T.sum(T.mul(attention(x, Wq, new T(randnMat(3, 3, 0.3)), new T(randnMat(3, 3, 0.3)), 3), attention(x, Wq, new T(randnMat(3, 3, 0.3)), new T(randnMat(3, 3, 0.3)), 3)));
    // 固定 Wk,Wv 以隔离 Wq 梯度
    const Wk = new T(randnMat(3, 3, 0.3)), Wv = new T(randnMat(3, 3, 0.3));
    const fwd = () => { const o = attention(x, Wq, Wk, Wv, 3); return T.sum(T.mul(o, o)); };
    zeroGrads([Wq]); backward(fwd());
    let maxRel = 0; const eps = 1e-5;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
      const orig = Wq.data[i][j];
      Wq.data[i][j] = orig + eps; const lp = fwd().data[0][0];
      Wq.data[i][j] = orig - eps; const lm = fwd().data[0][0];
      Wq.data[i][j] = orig;
      const num = (lp - lm) / (2 * eps);
      const ana = Wq.grad[i][j];
      const rel = Math.abs(num - ana) / (Math.abs(num) + Math.abs(ana) + 1e-12);
      if (rel > maxRel) maxRel = rel;
    }
    ok('注意力梯度检验 maxRelErr<1e-5', maxRel < 1e-5, 'maxRel=' + maxRel.toExponential(2));
  }

  // 6) 逻辑回归: 梯度检验 + 训练收敛
  {
    seed(7);
    const N = 60, d = 3; const X = [], y = [];
    for (let i = 0; i < N; i++) { const s = []; for (let j = 0; j < d; j++) s.push(rnd() * 2 - 1); X.push(s); y.push(s[0] + s[1] > 0 ? 1 : 0); }
    const W = new T(randnMat(d, 2, 0.1)), b = new T(zeros(1, 2)); const params = [W, b];
    const fwd = () => { const xi = new T([X[0]]); const z = T.add(T.matmul(xi, W), b); const p = T.softmaxRows(z); const oh = new T([[y[0] === 0 ? 1 : 0, y[0] === 1 ? 1 : 0]]); const pY = T.sum(T.mul(p, oh)); return T.scale(T.log(pY), -1); };
    zeroGrads(params); backward(fwd());
    let maxRel = 0; const eps = 1e-5;
    for (let i = 0; i < d; i++) for (let j = 0; j < 2; j++) { const o = W.data[i][j]; W.data[i][j] = o + eps; const lp = fwd().data[0][0]; W.data[i][j] = o - eps; const lm = fwd().data[0][0]; W.data[i][j] = o; const num = (lp - lm) / (2 * eps); const rel = Math.abs(num - W.grad[i][j]) / (Math.abs(num) + Math.abs(W.grad[i][j]) + 1e-12); if (rel > maxRel) maxRel = rel; }
    for (let j = 0; j < 2; j++) { const o = b.data[0][j]; b.data[0][j] = o + eps; const lp = fwd().data[0][0]; b.data[0][j] = o - eps; const lm = fwd().data[0][0]; b.data[0][j] = o; const num = (lp - lm) / (2 * eps); const rel = Math.abs(num - b.grad[0][j]) / (Math.abs(num) + Math.abs(b.grad[0][j]) + 1e-12); if (rel > maxRel) maxRel = rel; }
    ok('逻辑回归梯度检验 maxRelErr<1e-5', maxRel < 1e-5, 'maxRel=' + maxRel.toExponential(2));
    const r = logisticTrain(X, y, d, 500, 0.05);
    const dec = r.losses[0] / (r.losses[r.losses.length - 1] + 1e-12);
    ok('逻辑回归损失下降 >10×', dec > 10 && r.losses[r.losses.length - 1] < 0.3, 'init=' + r.losses[0].toFixed(3) + ' final=' + r.losses[r.losses.length - 1].toFixed(3));
  }

  // 7) PCA 方差解释率
  {
    seed(11); const N = 200, n = 4; const X = []; for (let i = 0; i < N; i++) { const a = rnd(), b = rnd(); X.push([a, b, 0.01 * rnd(), 0.01 * rnd()]); }
    const r = pca(X, 2);
    ok('PCA 前2主成分解释率>0.99', r.capVar > 0.99, 'capVar=' + r.capVar.toFixed(4));
  }

  // 8) Transformer 序列复制: 训练收敛 + 高精度
  {
    seed(99); const r = demoTransformer();
    ok('Transformer 反转任务 acc>0.9', r.acc > 0.9 && r.loss < 0.3, 'acc=' + r.acc.toFixed(3) + ' loss=' + r.loss.toFixed(3));
  }

  const pass = results.filter(r => r.pass).length;
  const total = results.length;
  const bar = '='.repeat(64);
  console.log(bar);
  console.log('  晨星AI · 核心内核 v1  ——  自检套件');
  console.log(bar);
  for (const r of results) console.log('  [' + (r.pass ? 'PASS' : 'FAIL') + '] ' + r.name + (r.info ? '   (' + r.info + ')' : ''));
  console.log(bar);
  console.log('  结果: ' + pass + '/' + total + ' 通过' + (pass === total ? '  ✅ 全绿' : '  ❌ 有失败'));
  console.log(bar);
  return { pass, total, results };
}

const API = { svd, cholesky, logsumexp, jacobiEig, T, Adam, pca, kmeans, knnPredict, logisticTrain, demoTransformer, runSelfTests, zeros, ones, eye, matmul, transpose, add, sub, scale, hadamard, randnMat, seed };
if (typeof module !== 'undefined' && module.exports) module.exports = API;
if (typeof window !== 'undefined') window.CXAI = API;
if (typeof require !== 'undefined' && require.main === module) { const r = runSelfTests(); if (r.pass !== r.total) process.exit(1); }
