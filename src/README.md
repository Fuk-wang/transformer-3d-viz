# 源文件（分片）

根目录的 `index.html` 是构建产物（单文件、零外链，Pages 直接发它）。
这里放的是构成它的所有分片，按下面顺序拼接即可还原，不需要任何构建工具。

```bash
cd src
bash build3.sh        # 拼出 ../index.html，并跑一次无头 Chrome 自检
```

拼接顺序（也是 build3.sh 里的顺序）：

| 文件 | 内容 |
|---|---|
| `v3-1.html` | `<head>`、全部 CSS |
| `v3-2.html` | `<body>`、HUD、控制面板、右侧卡片、帮助浮层 |
| `v5-real.html` | **真实 GPT-2 权重**：`REAL_ATTN`（12 层 × 12 头 × 5×5，以及接一个词后的 6×6）、每层首词占比、真实下一词分布 |
| `v3-3.html` | 常量、`attention()` 查表、`probDist()`、12 步脚本 `STAGES` |
| `v3-4.html` | WebGL2 基础设施：着色器、矩阵、`box` / `beam` / `quad` |
| `v3-4b.html` | `ribbon()`、球体、藻井地纹等几何工具 |
| `v4-stage.html` | 「长廊/展柜」视图：门 = 层、珠 = token 向量、12 条头轨 |
| `v3-6.html` | 主渲染器、状态机、相机、框图视图、UI 同步、URL 参数 |

## 数据是真的，怎么来的

`v5-real.html` 里的数字不是构造的，是本地对 **GPT-2 small（124M，12 层 / 12 头 / d=768）**
跑的一次完整前向：

- 权重用 `np.frombuffer` 直接读 `model.safetensors`（8 字节头长度 + JSON 头 + 裸 float32），
  前向只用 LayerNorm / 矩阵乘 / softmax / GELU —— numpy 就够，不需要 torch。
- 交叉验证：`New York is a city in the United States of` → `America` 0.9566、
  `1, 2, 3, 4, 5,` → `6` 0.9095、因果掩码右上角恒为 0。
- 存 5 位小数，逐行求和偏差 ≤ 2e-5。

重新导出用的脚本在 `tools/gen_real2.py`（需要同目录下有 `model.safetensors`、
`tokenizer.json` 和 `gpt2_numpy.py`，这些都太大、没进仓库）。

## 几个约束（改的时候别踩）

- 演示句必须是 **英文且切出 5 个 token**（`GPT is a language`）。中文在 GPT-2 的字节级 BPE 下
  是 20 个 token，每个汉字碎成 2–3 个字节片段，5×5 的画面就没了。
- 头号合法值是 0–11，没有「哨兵」值；不要再出现 `S.head === 4` 这种写法。
- 页面必须保持单文件、零外链、断网可打开。
