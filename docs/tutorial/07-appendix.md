# 07 · 附录：术语表与 FAQ

> 用法：**术语表**按主题分组，每条给"一句话解释 + 工程类比 + 首次详解的章节"；**FAQ** 收录阅读/运行本项目时最可能撞上的问题，每条给出诊断路径。

---

## A. 术语表

### 张量与形状

| 术语 | 一句话解释 | 类比 / 详见 |
| --- | --- | --- |
| tensor（张量） | 带设备属性的多维数组，PyTorch 的基本数据单位 | "会算梯度的 numpy 数组"，00 章 |
| shape（形状） | 每个维度的大小，如 `(B, N, 301)` | 函数的"类型签名"，全篇主线 |
| batch（批）/ `B` | 一次同时处理的样本数；shape 的第一维 | 请求的并发数，00 章 |
| padding（补齐） | 变长数据补占位对齐成统一形状 | 结构体数组按最大长度对齐，04 章 |
| mask（掩码） | 标记"哪些位置是真的"的布尔张量 | 位图/有效性标志，04 章 |
| dtype（数值类型） | fp32 / fp16 / uint8 等，决定精度与带宽 | 定点/浮点打包，02 章 |
| device（设备） | `cpu` / `cuda`，张量所在的物理位置 | NUMA 节点/协处理器内存，00 章 |
| broadcast（广播） | 小形状自动扩展成大形状参与运算 | 标量对向量的隐式 map，03 章 |
| scatter（散射） | 按索引表把值分桶累加/平均 | `groupby-agg` 的张量原语，03 章 |
| pooling（池化） | 把一组向量压成一个向量（mean/max） | 聚合函数，03 章 |
| embedding（嵌入） | 类别编号 → 可学习向量的查找表 | `enum → struct` 的可训练映射，00 章 |
| one-hot | 只有一维为 1 其余为 0 的类别编码 | 穷举枚举位，00 章 |

### 网络结构

| 术语 | 一句话解释 | 类比 / 详见 |
| --- | --- | --- |
| Linear（线性层） | `y = Wx + b`，参数就是 W 和 b | 一次矩阵乘法 + 偏置，00 章 |
| ReLU / 激活函数 | 逐元素非线性（负数清零） | 打破"多层=一层"的退化，00 章 |
| MLP（多层感知机） | Linear+ReLU 反复堆叠 | 最通用的可学习函数拟合器，00 章 |
| head（输出头） | 从共享表示分出的专用小输出 | 同一底层服务上的多个 endpoint，03 章 |
| trunk / backbone | 共享的中间表示生成部分 | 公共的 service 层，03 章 |
| logits | 未归一化的原始分数（softmax 的输入） | 打分阶段的中间值，00 章 |
| softmax | 任意分数 → 合法概率分布 | `exp` 归一化，00 章 |
| log_softmax | softmax 取 log（数值更稳，配 CE 用） | 00 章 |
| FiLM | 用状态算出 (γ,β)，对动作向量做 `γ⊙a+β` 乘性调制 | 状态当作"均衡器旋钮组"，03 章 |
| GNN / message passing | 信息沿图的边在节点间逐轮流动 | 邻居 gossip 协议跑 3 轮，03 章 |
| parameter vs buffer | 可学习参数 vs 随模型走的常量 | 可变配置 vs 只读资源，03 章 |

### 训练范式

| 术语 | 一句话解释 | 类比 / 详见 |
| --- | --- | --- |
| loss（损失） | "这次错多少"的标量打分函数 | 单元测试失败计数，00 章 |
| gradient（梯度） | 每个参数"往哪调能降 loss"的方向 | 调参方向指示器，00 章 |
| backpropagation（反向传播） | 一次自动算出所有参数的梯度 | 自动微分，`backward()` 一行，00 章 |
| learning rate（学习率） | 每次参数更新的步长 | 步进电机的步距，00 章 |
| epoch（遍） | 全部训练数据完整过一遍 | 全量数据的一次 ETL，00 章 |
| imitation learning（模仿学习） | 拿现成"老师"的选择当标准答案 | 抄作业暖启动，01 章 |
| warm-start（暖启动） | 先用便宜数据把网络拉到"会玩"再精修 | 预热缓存再上真实流量，01 章 |
| self-play（自我对弈） | 网络自己下棋自己出题 | 系统压测自己生成流量，05 章 |
| auxiliary task（辅助任务） | 顺带学一个相关小任务，帮助主表示 | 顺带埋的业务埋点反哺主逻辑，03/04 章 |
| overfitting（过拟合） | 死记训练数据、泛化变差 | 单元测试全绿但线上翻车，00 章 |
| replay buffer（经验回放） | 池化历史样本反复有放回抽样训练 | 消息队列的消费重放，04 章 |
| top-k 命中率 | 标准答案落进网络前 k 名的比例 | 排序质量指标 (recall@k)，04 章 |
| entropy（熵） | 预测分布的"犹豫程度" | 负载均衡的分散度，04 章 |

### 优化与数值稳定

| 术语 | 一句话解释 | 类比 / 详见 |
| --- | --- | --- |
| optimizer（优化器） | 按梯度更新参数的策略 | 调参执行器，00 章 |
| AdamW | 自适应步长 + 动量的优化器，工业默认 | 记住每个旋钮近期历史的智能调节器，00/04 章 |
| momentum（动量） | 平滑近期梯度方向，减少震荡 | 移动平均滤波，00 章 |
| weight decay / L2 | 惩罚大参数，防过拟合 | "旋钮别拧太大"的软约束，00/04 章 |
| scheduler（调度器） | 随训练进程调整学习率 | 变步长搜索，04 章 |
| cosine annealing（余弦退火） | 学习率沿余弦曲线从大到小 | 04 章 |
| grad clip（梯度裁剪） | 梯度范数超限就等比压回 | 保险丝/断路器，04 章 |
| AMP / autocast | 前向自动混用 fp16 提速省显存 | 变精度计算，04 章 |
| GradScaler | loss 放大再缩小，防 fp16 梯度下溢；顺带跳过坏步 | 浮点定标 + 脏数据跳过，04 章 |
| NaN / inf 巡检 | 周期检查梯度和参数是否有限 | 健康探针，04 章 |

### MCTS 与搜索

| 术语 | 一句话解释 | 类比 / 详见 |
| --- | --- | --- |
| MCTS | 用大量"模拟对局"修正网络直觉的树搜索 | 蒙特卡洛压测，05 章 |
| simulation（模拟） | 选择→扩展→评估→回传的一次循环 | 一次探路，05 章 |
| PUCT | 选择分支的公式：`Q + c·P·√N/(1+n)` | 利用与探索的加权调度，05 章 |
| Q value | 某分支历史模拟的平均评分 | 该路径的 SLA 平均值，05 章 |
| visit count（访问数） | 某分支被模拟过的次数 | 采样次数；分布本身就是训练目标，05 章 |
| prior（先验） | 网络 policy 给的初始打分 | 冷启动评分，05 章 |
| ISMCTS | 信息集 MCTS：每次模拟补全隐藏信息 | 对每个"可能世界"分别压测取平均，05 章 |
| determinize | 把隐藏信息随机补全成一个确定局面 | 猜对手手牌的开局采样，05 章 |
| Dirichlet noise | self-play 时混进根节点先验的随机噪声 | 强制探索的流量染色，05 章 |
| temperature（温度） | 采样随机度：1.0 按比例，→0 贪心 | softmax 的浓度旋钮，05 章 |
| candidate shortlist | 扩展时只挂生成器挑的前几个候选 | 只压测灰度候选集，05 章 |

### 工程机制

| 术语 | 一句话解释 | 类比 / 详见 |
| --- | --- | --- |
| checkpoint（检查点） | 模型+优化器+调度器的可恢复快照 | 事务日志/WAL，04 章 |
| state_dict | PyTorch 对象的序列化字典表示 | 内存转储格式，04 章 |
| schema version | 特征布局版本号，checkpoint 加载强校验 | API 版本门禁，02 章 |
| atomic write（原子写） | 写临时文件再 `os.replace` 改名 | 蓝绿发布/原子 rename，04 章 |
| shard | 数据切分落盘的独立单元 | 分片/分区分段，01 章 |
| materialize（物化） | 从 snapshot 延迟重建完整候选矩阵 | 惰性求值 + 按需物化视图，01 章 |
| spawn | Windows 必用的子进程启动方式（重新 import） | 独立进程冷启动，01 章 |
| 攒批（batched inference） | 攒一批请求一次推理 | 微批处理，05 章 |
| H2D | 主机内存 → 显存的拷贝（昂贵，省着用） | 跨 NUMA 搬运，02 章 |

---

## B. FAQ

### B1. 报错 `... must have shape (B,N,301)` / 各种 shape mismatch

**诊断路径**：打开 03 章的"形状合同"表，从报错的那一行往上找——几乎所有 shape 错误都源于：① 忘了 batch 维（单样本调用没 `unsqueeze(0)`，`net.forward` 入口有兼容但自建 batch 时容易漏）；② 候选集没 padding 直接 stack（变长 N 不一致）；③ 用了旧 schema 的数据喂新网络（02 章 schema 门禁会先拦截大部分这种情况）。

### B2. loss 变 NaN / 报 `FloatingPointError`

NaN 是数值训练的头号事故。本项目的防线（04 章）会在三个位置提前炸掉：loss 非有限（backward 前）、梯度非有限（深检）、参数非有限（深检）。排查顺序：

1. **学习率过大**（最常见）：lr 减半重试；
2. **脏目标**：数据里有 NaN/异常值（01 章的物化校验和 06 章的质量过滤就是为这个设的）；
3. **inf × 0**：检查是否绕过了 mask（04 章 policy loss 里 `masked_fill` 的注释）；
4. fp16 溢出：确认 `GradScaler` 启用、`--amp` 没被关；
5. 仍复现 → `finiteness_check_interval=1` 让深检每步跑，用第一个炸掉的 step 定位 batch。

### B3. CUDA out of memory

显存大头是**候选矩阵 + 它引发的激活**（04 章 4.5）：`batch × max_N × 301`。按顺序试：`--batch` 减半 → `--max-candidate-batch` 减半（65536 → 32768）→ 关 `--amp` 对照（确认不是 fp16 慢性碎片）。装箱器保证单个巨无霸只影响它所在的块，所以调预算不会拖累全批吞吐。

### B4. loss 不下降 / policy top1 很低

1. 先看 04 章 4.8 的指标族：top1 低但 top3 高 = 排序对、自信心问题；top1/top3 全低 = 真没学会；
2. 检查学习率（太大震荡、太小太慢——00 章 toy 脚本的练习就是为这个设计的体感）；
3. 检查数据：样本里 policy 是否全零/等价类摊平是否正常（01 章练习的打印脚本）；
4. 全合法集候选数极大（p95 近 2000），top1 基线本身就低——用"随 epoch 的**趋势**"而不是绝对值下结论（06 章指标行解读）；
5. loss 降到平台不一定是问题：policy CE 有理论下限（目标分布自身的熵），判断"该停了"还是"贴地板"的方法见 04 章 4.10。

### B5. top1 不低，但 benchmark 赢不了启发式

打分准 ≠ 棋力强，中间隔着：① value 头的质量（MCTS 评估靠它）；② 候选生成器覆盖（好动作不在短列表里，搜索永远看不见——05 章 5.3 的头号风险）；③ 模拟数（60 次对几百候选的搜索很浅）；④ 评测噪声（20 局粒度 5%，差 10 个百分点以内不算数——05 章 5.5）。

### B6. 加载 checkpoint 报 schema 版本 / "missing trainer state"

两类门禁（04 章 4.9）：schema 不匹配 = 特征编码改版了，旧权重作废，重新训练；missing trainer state = 旧格式 checkpoint 没存 optimizer 等，不能 `--resume`，删掉重训。

### B7. Windows 下子进程报错 / 卡住

spawn 会重新 import 脚本：入口必须走 `python/bootstrap_imitation.py` 脚本方式（不要 REPL 调 main）；worker 里的重 import 故意延迟到函数内（bootstrap 顶部注释）；BLAS 线程数被显式压成 1（01 章 1.7），别"优化"掉这几行。

### B8. 系统内存（RAM）吃满

三个可能的蓄水池：生成阶段在飞的 shard 结果、训练阶段物化池的候选矩阵、`Executor.map` 式的全量提交（本项目已改为限流调度，01 章 1.7）。调小 `--workers` / `--materialize-workers`；确认没绕过 `generate_imitation_sample_shards` 直接用内存版 `generate_imitation_samples` 生成大数据集。

### B9. 为什么没有"纯 AlphaZero self-play"的训练入口？

有零件没总装：`train.py` 的 `run_loop` 是参考实现（05 章闭环的最小代码化），`mp_selfplay.SelfPlayPool` 是工业化的数据生成端，但**当前没有顶层入口把它们串成一键训练**——bootstrap 是目前唯一维护的端到端入口（06 章 6.7）。这也是参与本项目的最佳切入点之一。

### B10. 我应该按什么顺序读代码？

教程章节即阅读顺序：`net.py`（03 章）→ `train.py`（04 章）→ `selfplay.py`（01/05 章）→ `rust_mcts.py` + `evaluate.py`（05 章）→ `bootstrap_imitation.py`（06 章）。`hierarchical_policy.py` 按需查（01 章 1.4/1.5、02 章 2.4 的引用点都标了函数名）；`mp_selfplay.py` / `replay_worker.py` 属下一阶段（06 章 6.7）。配套的权威参考：`docs/ai-python-code-map.md`（API 字典）、`docs/ai-action-encoding.md`（编码规范）。

---

## C. 延伸阅读

读完本教程想继续深挖时的经典材料（按对本项目的相关度排序）：

1. **AlphaZero 原理**：Silver et al., *A general reinforcement learning algorithm that masters chess, shogi and Go through self-play*（Science, 2018）——本项目的训练范式直接源于此：MCTS visit 分布作 policy 目标、value 头终局预测、自我对弈闭环。
2. **ISMCTS**：Cowling, Powley & Whitehouse, *Information Set Monte Carlo Tree Search*（IEEE TCIAIG, 2012）——手牌隐藏信息下搜索的正确姿势（05 章的 determinize）。
3. **PyTorch 入门**：官方教程 *PyTorch 60 Minute Blitz*（pytorch.org/tutorials）——00 章内容的官方展开，动手向。
4. **AdamW**：Loshchilov & Hutter, *Decoupled Weight Decay Regularization*（ICLR, 2019）——理解"AdamW 的 weight decay 与显式 L2 不是一回事"（04 章两者的分工）。

项目内部参考（非历史文档）：`docs/architecture.md`（系统全景）、`docs/ai-action-encoding.md`（编码规范与设计权衡）、`docs/ai-python-code-map.md`（API 速查）、`docs/roadmap.md`（阶段规划）。
