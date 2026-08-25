# 神经算子 vs 固定token Transformer：Burgers 方程对照实验

一个可以在笔记本 CPU 上约 15 分钟跑完的「用神经网络解 PDE」最小对照，对应文章三个论点：

1. **分辨率泛化（token化问题）**：FNO 用谱卷积，与网格解耦，训练在 128 点，直接测 256/512 点；
   固定 token 的 Transformer 换分辨率必须插值，误差随分辨率劣化。
2. **粘性外推（多尺度问题）**：训练在 ν=0.1 的光滑解上，测试 ν=0.01 出现激波的解，两者误差都爆到 0.46。
3. **与传统数值方法对比**：FNO 推理比谱方法快约 67 倍，但误差 0.0049 vs 参考解 0。

## 环境准备

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib
# 国内网络可加镜像：-i https://pypi.tuna.tsinghua.edu.cn/simple
```

全程 CPU 即可，不需要 GPU。

## 运行顺序

```bash
python gen_burgers.py    # 谱方法生成数据（训练500/测试120 @128点 + 精细512 + 低粘性外推集），约 5 分钟
python train.py          # 训练 FNO 与 TokenTransformer（各约 1 分钟），存 data/fno.pt 与 data/transformer.pt
python evaluate.py       # 三组评估：分辨率泛化 / 粘性外推 / 速度精度，写 results/results.json
python plot_results.py   # 出图 results/fig_*.png
```

## 文件说明

| 文件 | 作用 |
|---|---|
| `gen_burgers.py` | 伪谱 RK4 求解 1D 粘性 Burgers 方程（u_t + u u_x = ν u_xx，周期边界），生成全部数据集 |
| `models.py` | FNO1d（谱卷积）+ TokenTransformer（学习位置编码、无坐标信息） |
| `train.py` | 固定种子训练两个模型，输出验证 MSE |
| `evaluate.py` | Exp A 分辨率泛化、Exp B 粘性外推、Exp C 与传统谱方法对比 |
| `plot_results.py` | 出图 |
| `make_formulas.py` | 公式渲染（微信不支持 LaTeX，公式一律出图） |
| `results/results.json` | 本次跑出的全部关键数字 |

## 关键实现细节（踩坑点）

- **谱方法参考解就是「传统数值方法」**：数据生成和 Exp C 的对比基线用同一个求解器，
  ν=0.1、128 点时 dt 由 CFL 与粘性稳定性限制 `ν k_max² dt < 1` 共同约束。
- **FNO 的分辨率不变性**：谱卷积在频域截断前 modes 个模，输出网格点数与权重无关，
  所以 128 点训完可以直接在任何分辨率上推理——这是它与固定 token 模型的本质差别。
- **TokenTransformer 故意做成「天真的 token 化」**：学习位置编码、不注入坐标，
  这正是科学数据 token 化最大的坑：把连续函数硬切成固定数量 token，分辨率一变就要插值。
- **外推测试集 ν=0.01**：同一批初始条件、更小的粘性，解在 t=1 前形成近激波，
  训练分布内从未见过这种多尺度结构。
- 所有脚本固定随机种子（20260825），结果可复现。

## 本次跑出的结果

见 `results/results.json`（复现时毫秒级浮点差异除外，量级稳定）。

## 开源

本实验代码将发布在 GitHub：https://github.com/aduo2233/fno-vs-token-transformer-burgers
（推送暂存目录见同 run 的 `repo-staging/`，发布前请先创建远程仓库。）
