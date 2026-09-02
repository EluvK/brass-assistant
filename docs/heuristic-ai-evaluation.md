# Heuristic AI 当前评价逻辑说明书

本文是对 `engine/src/ai/heuristic_ai` 当前源码行为的整理，服务于下一步“统一指标 + 阶段权重”的重构。本文描述的是代码现在实际做什么，不代表规则裁定，也不代表推荐的最终设计。规则仍以 `docs/brass-birmingham-rules.md` 和用户决定为准。

源码范围：`mod.rs`、`config.rs`、`context.rs`、`value.rs`、`board.rs`、`probability.rs`、`plan.rs`、`build.rs`、`network.rs`、`develop.rs`、`sell.rs`、`loan.rs`、`scout_pass.rs`、`cards.rs`、`lookahead.rs`。

## 1. 总体流程

`candidate_actions_k` 每次候选批次执行以下流程：

1. 根据当前状态创建 `EvalContext`，确定 `Phase`、剩余轮数、阶段换算权重。
2. 枚举合法 Build target，并一次性计算全手牌的 keep-score。
3. 计算当前生产计划 `Plan`。
4. 分别评价 Build、单铁路、双铁路、Develop、Sell、Loan、Scout、Pass。
5. 对 Build/Network/NetworkDouble 保留每种几何最多 `SOURCE_VARIANTS = 2` 个资源来源变体；Develop/Sell/Loan/Scout/Pass 通常各保留单个或少数计划。
6. 对大多数动作以 `score` 排序；动作中的具体 card index 由独立的牌评分决定。
7. 去重时使用 `operation_key`，刻意排除 card index，因此同一个操作不会因换一张牌变成多个操作节点。
8. `choose_action` 还会对候选做确定性的 2-ply lookahead，并将第二个自己的动作以 `alpha` 折算后加到第一动作分数上。

候选动作的 `Decision` 有三个值：

|字段|含义|
|---|---|
|`mv`|一个可执行的具体动作，包含资源来源和一张/三张牌的引用|
|`score`|操作本身的评价，不应理解为某张牌的评价|
|`card_score`|被消耗牌的 keep-score 总和，仅用于记录/辅助选择，当前不加进 `score`|

## 2. 统一货币：ScoreParts

当前 `ScoreParts` 有六个指标：

|指标|当前语义|`total` 中的处理|
|---|---|---|
|`vp`|即时或预期 VP、连接图标、Develop 解锁目标的抽象价值|原值相加|
|`money`|现金变化，花钱为负、卖资源得到钱为正|乘 `money_w`|
|`income`|收入等级变化或翻面带来的收入|乘 `income_w`|
|`flex`|手牌可行动性的变化，当前主要是 Network 新增可用牌|乘固定 `cfg.value.flex`|
|`strategic`|无法直接归入 VP/现金/收入/牌灵活性的战略奖励，如商家、市场热度、规划、节奏|原值相加|
|`risk`|风险/机会成本/不安全状态，约定通常存负数|原值相加|

公式为：

```text
total = vp
      + money * money_w(phase)
      + income * income_w(phase)
      + flex * flex_w
      + strategic
      + risk
```

默认基础换算：`vp = 1.0`、`£1 = 0.12`、`1 income = 0.25`、`flex = 0.8`。注意 `strategic` 和 `risk` 当前没有统一的阶段乘数；它们虽然放在统一结构中，实际仍由各操作的系数直接决定。

### 阶段与时间换算

阶段由 `era_phase` 分为：Canal-Early、Canal-Late、Rail-Early、Rail-Late。`era_frac = rounds_remaining / 8`，限制在 0 到 1。

|阶段|income_w|money_w|network_w|2-ply alpha|endgame 窗口|
|---|---:|---:|---:|---:|---:|
|Canal-Early|`0.25*(1.8+0.6*era_frac)`|`0.12*0.55`|0.1|0.6|最后 2 轮|
|Canal-Late|同上|同上|0.1|0.6|最后 2 轮|
|Rail-Early|`0.25*(1.2+0.5*era_frac)`|`0.12*0.8`|1.0|0.6|最后 1 轮|
|Rail-Late|0|`0.12*5/3`|0.85|0.35|最后 1 轮|

未来价值折扣为 `0.3 + 0.5 * era_frac`，用于连接的未来图标，以及部分“现在行动、以后兑现”的价值。

## 3. Build 评价

入口是 `score_build_candidate`，最终由以下部分组成：

```text
vp       = tile.vp * flip_probability + 已有相邻自家连接图标价值
income   = tile.income * flip_probability
money    = -建造总成本
strategic = 自给 + 扩张 + 铁路煤短缺 + 成本效率
          + 资源市场价值
          + 啤酒经济
          + 免费资源占比奖励
          + 流派奖励
          + Rail-Late 有酒收尾奖励
risk     = 自家覆盖损失 + 卖不掉/留在板块上的资源损失
```

### 3.1 先决条件和硬性/软性淘汰

- 没有对应的下一张行业板块：返回 `risk = -∞`。
- 配置禁止时，Canal level-1 Brewery：返回 `risk = -∞`。
- 现金不足：不直接淘汰，而是给 `-(cost-cash) * unaffordable_per_pound` 的风险；贷款仍可能通过其他路径解决。

### 3.2 翻面概率 `flip_probability`

这是 Build 评价中最重要的隐含模型，范围最终限制到 `[0.05, 0.9]`。

资源厂（Coal/Iron）：

- 有合法市场销售且放置后能卖完全部资源：直接使用 `sellout = 0.9`。
- 否则按时代需求 + 市场稀缺度计算；Coal 在 Rail 为 `0.85` 基础需求，Canal 为 `0.55`；Iron 分别为 `0.5/0.4`。
- Island Coal（不能到达接受 Coal 的商家）单独处理：Canal 为 `base + 煤价热度 * bonus`，上限 0.4；Rail 为另一套更高的 base/上限 0.9。
- 市场越稀缺，资源被消耗的概率越高。

Brewery：

- 需求 = 自己未翻面的可售产业所需啤酒；Rail 再加铁路啤酒需求 buffer 1.0。
- 供给 = 已有自家未翻 Brewery 的桶数 + 当前下一座 Brewery 的桶数。
- Canal 且需求很低：使用 `brewery_canal_no_demand`。
- 供给大于需求：使用 surplus 概率；否则使用 satisfied 概率。

可售产业（Cotton/Goods/Pottery）：

- 只有在可到达接受该产业的商家并且有啤酒时才真正有较高翻面概率。
- 具体地点存在商家：根据“商家+有酒 / 只有商家”加不同奖励；没有商家则加负值。
- 相邻有未建铁路：加“以后可能接通商家”的奖励。
- 手牌过少增加风险：0 张、1 张、2–3 张分别扣 10、5、2；4 张以上不扣。
- 计划级（无具体地点）则仅判断全图有没有接受该产业的商家，以及自家/商家是否有啤酒，使用 `plan_no_merchant / plan_no_beer / plan_ready`。

### 3.3 产业通用指标

- `link_self_value`：若地点旁已有自家连接，板块的 `link_vp * flip_prob * 0.5` 计入 VP。
- `self_sufficiency`：Coal/Iron 每个产出资源桶加 0.15；其他产业为 0。
- `expansion`：地点相邻每条未建连接加 0.1。
- `cost_efficiency`：`(income + vp) / cost`，上限 2.0。它是战略分而不是 VP。
- `free_riding`：计算本次煤/铁需求中可由棋盘免费来源满足的比例；超过 0.5 的部分按 0.8 奖励。包括消耗对手资源并翻对手板块带来的间接收益，但动作分本身没有单独的“对手损失”指标。
- `own_overbuild`：覆盖自家板块时，扣除被覆盖板块打印 VP × 1.0；覆盖对手耗尽资源板块不扣该项。

### 3.4 资源厂市场价值

适用于 Coal/Iron：

- Coal 市场稀缺度完整计入，Iron 乘 0.6，避免过度生产 Iron。
- Coal 必须可到达接受 Coal 的商家才能走“放置即出售”路径；Iron 不需要商家连接。
- 可出售时，市场自动出售得到的现金放入 `money`。
- 现金再按 0.4 比例作为即时节奏奖励；全部卖完再加 sellout bonus 1.5。
- 煤价在 5–8 窗口内时，每卖出一桶额外加煤价热度奖励 1.9；Canal 再乘 1.25。
- 市场稀缺度 ×（1 + 实际卖出桶数）× 0.6 作为战略值。
- 未卖出的桶每桶扣 0.5；但 Rail Coal 不扣，因为后续铁路/建造会消耗。
- 未连通 Coal：Canal 固定扣 0.5；Rail 使用“稀缺度 × (1.2 + 0.25 × 桶数)”的投机价值。未连通 Iron 使用“稀缺度 × 1.2”的保底价值。
- Rail 且煤市场接近空时，所有合法 Coal Mine 另加紧急煤短缺奖励；按市场稀缺度、板块等级和桶数放大。

### 3.5 啤酒经济

- 可售产业：到达接受商家加 0.6；商家和足够啤酒都具备再加 0.8；否则加 `-0.3`，故缺酒不是硬淘汰。 
- Brewery：计算“已有自家啤酒 + 当前板块啤酒”与所有未翻可售产业需求的差额。
  - 有需求时基础支持 0.8，无需求时 0.4。
  - Rail 额外加 Brewery 价值 2.0。
  - 超过需求的每桶扣 0.6，防止无目标囤酒。

### 3.6 流派和后期条件

- `Plan` 选定的产业从 Canal-Late 开始建造时加 0.5；Canal-Early 不加。
- Rail-Late 的可售产业如果确实能找到自家或商家啤酒，额外加 1.2。

## 4. Network / 单铁路评价

`score_network_candidate` 的指标为：

```text
vp     = (当前连接图标 + future_discount * 空位未来图标) * network_w
flex   = 新接入的地点牌 * 0.6 + 手牌行业牌 * 0.1
money  = -(连接基础成本 + Rail 所需煤的估计成本)
strategic = 商家接入 + 探索 + 流派空位 + Rail-Early 啤酒农场锁
```

场景判断：

- Canal 需要已有 Canal link，Rail 需要已有 Rail link，否则该动作类型直接不出候选。
- 当前图标来自连接两端及 via-farm：商家固定 2 VP，翻面产业使用其 link VP。
- 未来图标按空城市槽计算；普通产业槽值 1，能放 Brewery 的槽值 2，空 Brewery farm 值 2。
- `hand_access_gain` 只把“本来不在网络、且本次两个端点之一是该地点”的 Location card 算新增访问；Industry card 只按手上总数粗略计入，不按具体产业判断。
- 探索奖励为 `max(1.6 - 0.3 * 当前时代剩余可建连接数, 0)`，剩余连接越少反而越高。
- Canal-Late、Rail 阶段若连接端点有流派产业的空槽且仍有该产业板块，给 0.5。
- Rail-Early 连接触及 Brewery farm，给 1.2。
- 任一端点为商家，给 1.5。

Rail 双铁路不是 `ScoreParts` 相加后保留分解，而是：两个单铁路 `total` 相加，扣除双铁路附加成本换算后的 surcharge，再加动作节奏和啤酒农场锁奖励。

- Rail-Early 双铁路节奏奖励 1.2，其他阶段 0.6。
- 任一铁路触及 Brewery farm，额外加 0.8。
- 资源来源变体优先列出不同的 Coal 来源；双铁路还优先自家啤酒来源。不同来源的动作可能产生不同翻面结果，但分数通常仍共享同一个几何分数。

## 5. Develop 评价

Develop 当前不是直接评价实际移除结果，而是先评价“移除后解锁的目标板块”。

单个目标的抽象价值：

```text
base = rail_era_tile(0.35) 或 canal_era_tile(0.12)
     + 被移除板块等级 * 0.18
     + 若下一张是 Rail tile，加 0.25
     + Canal-Early 的 level-1 Brewery 特殊奖励
     + Canal-Early Iron 价格阶梯奖励/惩罚
     + Canal 阶段加 0.15
     + 流派产业加 0.3
     + 手上有能建该产业的牌加 0.3
     - Brewery/Coal level>=2 的 guardrail 扣分
```

候选和组合判断：

- `can_develop`、有可用 Iron source、最便宜 Iron 可支付，是前置条件。
- 配置可直接禁止 Iron level 2+；也可禁止 Canal-Early Brewery level 2+。
- 按目标抽象价值排序，最多输出 `SOURCE_VARIANTS` 个不同的第一移除产业。
- 如果有第二个 Iron source 且可支付，在“其他最高目标”和“同产业下一张目标”中选价值较高者。
- 第二个目标只按 `second_target_scale = 0.4` 计入。
- `money = -实际 Iron 成本`；非免费 Iron 另在 `risk` 扣 0.6。
- Canal 阶段整个 `vp` 乘 2.0；单开发再扣 2.0，双开发加 0.5。
- 本时代已经 Develop 的次数超过限制后，按 `over² * 2.0 + over` 扣风险；Canal 限制 4，Rail 限制 1。
- 具体动作选择最优 Iron source 和 keep-score 最低的一张牌。当前分数没有单独表示“消耗哪种产业”“解锁目标可达性”等通用指标。

## 6. Sell 评价

Sell 先枚举合法 Sell targets，然后做一个“尽量卖全部”的贪心计划；每次试加一个目标都会复制状态并实际执行模拟，只有能让所有计划中的板块翻面才接受。

### 6.1 目标和路线排序

- 每个目标先计算可用商家路线。
- 有商家啤酒的路线，其商家 bonus 取最高者；按商家 bonus 降序，再按板块 `vp + income * 0.3` 降序。
- 路线内部按“使用商家啤酒得到的 bonus”降序。
- 输出多个变体时，仅对排序第一块尝试不同的首选路线起点，后续目标仍用首个可执行路线。
- 最终 Sell 的 keys 和对应 route 按升序 key 排列，以匹配 canonical action 编码。

### 6.2 Sell 的 ScoreParts

对每个实际卖掉的板块：

- `vp += printed VP`。
- `income += printed income`。
- 使用商家啤酒时，商家 bonus 转换为对应指标：VP bonus 进 `vp`，钱进 `money`，收入进 `income`，免费 Develop 进 `strategic`（默认 0.5）。
- 所有 income 最后乘 `1 + income_stream_share = 1.5`，模拟收入的持续现金流。
- `vp` 乘 `0.1 + 0.5 * (1 - era_frac)`：越接近时代结束越重视已兑现 VP，早期更重收入。
- 最后阶段若处于 endgame，加 urgency 3.0；否则 Rail-Late 加 baseline 1.2。

这里没有 Sell 的额外 `money` 基础收益；现金主要来自商家 money bonus，普通翻面收益体现在 VP/income。

## 7. Loan 评价

Loan 只有在引擎允许贷款时产生。它不是单纯“借钱”的固定价值，而是估计借钱后能否买到更好的动作：

```text
income   = -3 income levels
strategic = best_affordable_build_score(借钱后) - best_affordable_build_score(现在)
         + 同回合后续动作价值 * combo_scale
         + 低现金 idle bonus
         + 从“无可负担好建造”变成“有好建造”的 unlock bonus
         + Canal-Early startup bonus
         + Canal-Late 有近期可翻产业且现金不足的 bonus
risk     = 借款后收入跌入负收入/低收入区间的分层惩罚
         + 当前现金已经很多时的 rich penalty
```

具体场景：

- `after` 和 `now` 只遍历当前合法 Build target，并用 Build scorer 评价可负担的最高分；没有可负担建造时按 0 处理。
- 现金低于 24 且剩余轮数超过 1.5 时，模拟 Loan 后若本玩家仍能获得下一行动，将下一批候选最高分的正值按 0.7 加入。
- 现金低于 18，加 idle bonus 2.0。
- `now <= 0` 且 `after > 0.8`，加 unlock bonus 3.2。
- Canal-Early 前两轮加 startup bonus；现金低于 18 时为 6.0，否则 0.5。
- Canal-Late 第 6 轮以后，若有合法 Sell target 或已有未翻可售产业且现金低于 30，加 1.8；现金低于 18 时改为 2.8。
- 贷款后收入 `<= 0/-4/-7` 分别进入 breakeven/debt/deep-debt 风险档，最高扣 7.0。
- 当前现金 `>=30/42/55` 分别扣 1.0/2.4/5.0。

因此 Loan 依赖 Build scorer，是当前“操作评分嵌套操作评分”的典型例子。

## 8. Scout 与 Pass

这两个操作尚未使用 `ScoreParts`，直接产生标量 `score`。

### Scout

前置条件是 `can_scout` 且手牌至少三张。手牌按 keep-score 从低到高取前三张弃掉：

```text
score = dead_count * 0.96
      - (3 - dead_count) * 0.48
      + hand_refresh_score
```

其中 `dead_count` 是前三张中 keep-score `<= 0` 的数量。`hand_refresh_score` 只看弃牌后的保留牌：

```text
5.0
* 低价值保留牌比例(score <= 1.0)
* (0.35 + 0.65 * 高价值牌缺口比例)
* 平均质量缺口
```

高价值牌是 keep-score `>= 1.8`，理想数量为 2；平均质量缺口以 location card 基础 keep-score 1.15 为锚点并限制在 0–1。也就是说，保留牌越弱、越缺少高价值牌，Scout 越有价值；只有三张废牌而其余手牌很强时，不会给整个手牌刷新很高的奖励。

### Pass

正常 `score_pass_result` 固定为 0。它没有现金、收入、VP、flex 改变，因此排在正分行动之后、亏损行动之前。若候选完全为空，`pass_decision` 使用配置中的 fallback `-0.5`，并保留最低 keep-score 的牌引用。

## 9. 手牌评价：Card keep-score

牌评价和操作评价是两个独立维度。keep-score 越低，越适合被操作消耗；越高，越应保留。当前默认基础值：Location 1.15、Industry 1.0、Wild 3.8。

### 9.1 所有牌共有逻辑

- Location 的重复只按完全相同地点计数；Industry 不区分打印的产业组合，只按“Industry card 总数”视为同类重复。
- 重复牌从第二张开始每张扣 0.48，因此重复牌更适合消耗。
- Wild 不参与普通 duplicate count；但 Wild 多张时每多一张加 0.35，表示多张万能牌仍有保留价值/同时拥有的结构价值。

### 9.2 Location card

- 城市未满：统计该地点当前合法 Build target 数量，每个目标加 0.28，最多统计 3 个。
- 城市已满：
  - Rail 阶段若存在自家 Coal/Iron/Brewery 可升级到更高等级，且手中没有对应产业牌或万能产业牌，扣 0.45；保留地点牌作为资源升级入口。
  - 否则扣 1.05，表示该地点牌基本失效。
- 数据若出现非城市 Location，按保守的 full-useless penalty 处理，并触发 debug assertion。

### 9.3 Industry card

- 遍历该牌打印的产业角色，取其中合法 Build target 数量的最大值。
- 没有任何合法目标，扣 0.65。
- 有目标时按最大目标数 × 0.22，最多统计 3 个。
- Canal 阶段 Industry card 总数大于 1 时，再扣 0.22，表示重复产业牌在 Canal 灵活性较差。

### 9.4 Wild card

Wild 的核心仍是高基础 keep-score，因此通常最不愿意丢弃；多张 Wild 还会按额外 Wild 数加 0.35。它没有根据当前具体地点/产业需求做进一步细分。

### 9.5 牌如何进入动作

- Build 只能从 `valid_build_cards` 返回的匹配牌中选 keep-score 最低者；注释意图是优先普通匹配牌而不是 Wild，但当前实现实际使用 keep-score 最低排序，若 Wild 的最终 keep-score 更低仍可能被选。
- 其他普通动作（Network、Develop、Sell、Loan、Pass）使用全手牌排序的第一张。
- Scout 消耗排序后的前三张，`move_card_score` 是三张分数之和。
- 当前操作横向比较不加 `card_score`，所以“为了动作消耗一张高价值牌”的代价不会自动反映在操作 `score` 中。

## 10. Plan（流派/生产计划）评价

`compute_plan` 只在可售产业（Cotton/Goods/Pottery）中选择目标产业。每个产业的计划分为：

```text
计划数量 * 平均打印 VP
* plan_flip_probability
* beer_factor
* hand_factor
```

- 计划数量 = `min(剩余该产业板块数, 当前空的合法槽位数)`。
- `beer_factor`：已有啤酒足够则为 1；不足时为 `0.4 + 0.6 * own_beer / beer_needed`，限制不超过 1。
- `hand_factor = 0.5 + 0.25 * hand_support`，`hand_support` 统计能支持该产业的地点/产业/WildIndustry 牌，最多 3。
- 没有可行产业时默认 Cotton、数量 0。

计划结果只作为其他 scorer 的软条件：Build 和 Network 在非 Canal-Early 阶段有 plan bonus，Develop 有 plan bonus；它不是一个独立的 ScoreParts 指标。

## 11. 局面估值与 Lookahead

### 11.1 MCTS leaf `evaluate_position`

局面估值不是某个动作的 `ScoreParts`，而是：

```text
当前已结算 VP
+ 棋盘估计 VP（已翻板块 + 未翻板块*0.25 + 自家连接图标）
+ money * 当前 money_w
+ income * 3.0 * 当前 income_w
+ 全手牌 keep-score 总和 * 0.8
```

连接图标估计严格按当前结算规则：翻面产业使用 printed link VP，商家固定 2，农场也计入。未翻产业只取 25% 的打印 VP 作为期望。

### 11.2 2-ply

`choose_action` 对第一层候选逐一模拟，推进回合并处理时代结束；如果回合仍回到自己，再取第二动作最高分：

```text
value = first_score + alpha * max(second_score, 0)
       + end_of_turn_penalty
```

`alpha` 阶段值为 Canal 0.6、Rail-Early 0.6、Rail-Late 0.35。

回合结束现金低于 15 时可能扣安全风险。扣分由现金缺口比例、当前收入是否为负、Canal/Rail、以及剩余 runway 决定；本回合收入提升达到 2.5 时免除该惩罚。该惩罚发生在动作 `ScoreParts::total` 之外，所以最终选择分不再是纯粹的统一指标线性组合。

## 12. 当前逻辑中适合统一、以及尚未统一的内容

### 已经接近统一指标的部分

- 直接收益：VP、现金、收入。
- 资源成本：Build/Network/Develop 的钱都进入 `money`，但现金不足和贷款风险仍在操作内部处理。
- 兑现概率：Build 使用 flip probability；Sell 直接使用已执行成功的翻面结果；Network 使用当前/未来连接图标。
- 资源和啤酒供应：商家可达、啤酒可达、市场稀缺度、免费资源比例已经有公共查询函数。

### 仍然是操作专属指标或标量的部分

- `strategic` 混合了商家、市场热度、探索、规划、节奏、煤短缺、免费开发、手牌刷新等不同含义。
- `risk` 混合了不可负担、覆盖 VP、资源剩余、收入债务、Develop 次数超限等不同风险。
- Scout 完全是标量公式；Pass 是固定值。
- Develop 的“目标板块价值”并非实际 `ScoreParts`，先使用 `f64` 组合后塞入 `vp`。
- Loan 把 Build 最高分和下一动作最高分嵌入 `strategic`。
- Sell 的目标排序、路线选择和最终评分不是同一个指标体系：排序使用商家 bonus/`vp+income*0.3`，最终分又使用收入流和阶段 VP 缩放。
- card keep-score 不进入操作分；因此操作与所牺牲牌的机会成本分离。
- 2-ply 的第二动作和回合末现金惩罚在统一总分之外再次修改选择结果。

## 13. 面向下一次重构的指标候选表

以下是从当前逻辑抽出来、适合成为跨动作通用 `ScoreParts` 字段的候选语义。这里不要求一次全部实现，而是将当前专属逻辑映射到可横向比较的概念：

|建议通用指标|当前来源|
|---|---|
|即时 VP|Build printed VP、Sell 已翻 VP、商家 VP bonus|
|预期 VP/兑现概率|Build flip probability、Develop 解锁概率、未来连接图标|
|收入提升|Build/Sell 翻面收入、商家 income bonus|
|现金变化|Build/Network/Develop 成本、资源自动销售、商家 money bonus|
|行动灵活性|Network 新接入牌、Scout 手牌刷新、保留可建目标|
|资源供给价值|Coal/Iron 产出、市场稀缺、Rail 煤短缺、自给/免费资源|
|啤酒供给/啤酒锁|Brewery 需求匹配、Sell 有酒、Rail farm lock|
|商家/网络可达性|商家连接、卖出路线、未来开路、连接图标|
|产业发展潜力|空槽、扩张连接、Develop 解锁、Plan 对齐|
|行动效率/节奏|双铁路、双开发、2-ply 后续动作|
|终局紧迫度|Canal endgame 未翻消失、Rail-Late Sell、未来价值折扣|
|机会成本|覆盖自家板块、消耗高价值牌、Develop 次数、借贷收入损失|
|失败/风险|卖不掉资源、缺啤酒、现金安全线、负收入债务|

重构时尤其需要决定：哪些指标是“事实测量值”（例如可达商家数量、可翻 VP、资源桶数），哪些才是“阶段权重”；当前代码把二者混在 `strategic`/`risk` 和多个操作专属系数里，是评价函数复杂的主要原因。

## 14. 参考源码入口

- 统一结构与市场模型：`engine/src/ai/heuristic_ai/value.rs`
- 阶段换算：`engine/src/ai/heuristic_ai/context.rs`
- 所有默认参数：`engine/src/ai/heuristic_ai/config.rs`
- 动作汇总、候选去重、局面估值：`engine/src/ai/heuristic_ai/mod.rs`
- 操作评价：`build.rs`、`network.rs`、`develop.rs`、`sell.rs`、`loan.rs`、`scout_pass.rs`
- 共享事实查询：`board.rs`、`probability.rs`
- 手牌评价：`cards.rs`
- 流派和操作前瞻：`plan.rs`、`lookahead.rs`

