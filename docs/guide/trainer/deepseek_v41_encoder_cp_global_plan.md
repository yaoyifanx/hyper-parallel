# DeepSeek V4.1 ViT Encoder CP 跨 DP 解耦方案

## 先看一个心智模型：八张卡执行两份图文作业

把一次训练想成两份互不混淆的作业：DP0 负责图文样本 A，DP1 负责图文样本 B。
机器有 8 张卡，每份作业各占 4 张。**decoder 的分工**是固定的：DP0 的
rank `0..3` 消费 A 的文本 token，DP1 的 rank `4..7` 消费 B 的文本 token。
每组内部再按 `CP=2、TP=2` 切分，因此 decoder 的一个样本域只有 4 张卡。

**ViT 的分工可以另排。**若设置 `encoder_cp_size=8`，就让 8 张卡组成一个
ViT CP 组。它们共同处理 A 和 B：每张卡拿到 A 的一段 patch、B 的一段 patch，
在一次 packed ViT 调用中分别编码这两段。A、B 是两个独立 attention segment，
不会互相注意。rank 0 收齐两张图的 ViT hidden 并做 aligner，然后把 A 的
feature 送回 DP0 中需要它的 decoder rank，把 B 的 feature 送回 DP1 中需要它的
decoder rank。反向梯度沿相反路线返回 ViT。

```text
decoder 分组：  A 的文本 -> [0 1 2 3]     B 的文本 -> [4 5 6 7]
图像来源：    A 的 patch -> rank 0        B 的 patch -> rank 4
ViT 分组：                 [0 1 2 3 4 5 6 7]  (ECP=8)
                         每张卡各处理 A 的 1/8、B 的 1/8
feature 回传： A -> DP0 的 token owner     B -> DP1 的 token owner
```

同一批设备在 decoder 阶段和 ViT 阶段使用了**两套坐标**；decoder 的样本归属
保持不变。跨 DP 通信包括图像 patch、任务元数据、ViT attention 的 Q/K/V、
ViT hidden、结果 feature 及它们的反向梯度。改造前的代码把 ViT CP 组
放在 decoder 的四卡样本域里，所以无法配置 ECP8。要解除该限制，需要完成
“收集各 DP 的图像任务 → 跨 DP 分发 patch → ViT 编码 → 按原样本归还 feature”
这一整条路径。本文后面的 `encoder_bucket`（旧代码名为 `encoder_dp`）只是
ViT 的工作分桶，**不等于** DataLoader 的 DP 样本编号。

下面先列改造前的耦合，再给完整目标设计；第 8 节逐项说明已落地的较小改动。
第 5 节将本例展开到每个 rank 的 patch 数、feature 目的地和反向路径。

## 1. 目标与边界

目标是在 `pp_size=1` 的训练域内，让 ViT 的 `encoder_cp_size` 独立于 decoder 的
`cp_size × tp_size`。例如 `dp=2, cp=2, tp=2` 时，改造前的样本域最多只能选
`encoder_cp_size=4`；现已加入的 `global` 模式允许 `encoder_cp_size=8`，同时保持每个 DP 样本的
decoder token 仍由原来的 CP/TP rank 消费。

目标数据契约是把同一逻辑样本分别构造成 **decoder 输入**和 **ViT 输入**：前者按
LLM 的 DP/CP/TP 放置文本 token 与 loss 字段，后者按 encoder bucket/ECP 放置图像
patch；二者通过样本键和图像任务键关联。首版仍以 decoder DP 作为样本采样所有者，
但该所有者不等同于 ViT 工作桶。OmniBatch 应理解这两种布局，不能让同一个 `dp/cp`
参数同时代表两套模型策略。

这不是删去一条整除检查即可完成的修改。改造前 Omni 数据管线按 DP 分样本，仅在同一
DP 样本的 CP/TP rank 内复制图像字段；ViT 工作分桶、Ulysses 分组和 feature exchange
也使用该样本域。跨 DP 后，编码 rank 必须拿到其他 DP 样本的 patch，并把编码结果送回
原样本的 decoder。本文给出可实施的接口、通信顺序和验收方案。

约束如下：

- `group_scope=sample` 保持样本域分组和默认配置；`group_scope=global` 显式开启跨 DP 路径。
- 全局模式的工作域是 `device_mesh` 的 `(dp, cp, tp)` 全域，大小
  `W=dp_size × cp_size × tp_size`。均匀 Encoder-CP 分组要求 `W % encoder_cp_size == 0`。
- DeepSeek ViT 当前使用 Ulysses，Q/K/V 的 head 数都必须被 `encoder_cp_size` 整除。
  因此“任意范围”指不再受 decoder `cp × tp` 限制，并不意味着任意正整数都合法。
- 维持现有 `pp_size=1`、`micro_batch_size=1`、禁用 compile 和 placement validation
  的边界。ViT 在训练时按四层一组做 activation checkpoint，反向重算时恢复
  Encoder Attention 上下文。
- 当前 `sample` 和 `global` 模式都要求 `pixel_values` 不需要输入梯度。ViT、aligner
  和 decoder 参数的梯度必须完整可微。若以后需要输入像素梯度，补一个所有 rank 都
  参与反向的可微 patch transport，不能直接把有梯度的输入交给 eager 通信。

## 2. 改造前代码的耦合点

本节“现状”列及 §2.1、§2.2 的预取描述是**修复前基线**，用于解释 OOM 成因；
当前实现及未完成项见 §8。

| 文件与函数 | 现状 | 需要改变的原因 |
| --- | --- | --- |
| `distributed/encoder_parallel/runtime.py::EncoderParallelRuntime.__init__` | 父域固定为 `device_mesh[("cp", "tp")]`，`encoder_cp_size` 必须整除 `cp×tp` | 全局模式要从 `(dp,cp,tp)` 派生 Encoder-CP 网格 |
| `runtime.py::finalize_plan` | 只比较同一个 decoder parent 的图像元数据；对本 DP 样本分桶 | 跨 DP 要先形成统一的全域任务表 |
| `runtime.py::EncoderParallelPlan._execute_local` | `bridge.encode_local` 从本 rank 的 `pixel_values` 取 patch | Encoder-CP rank 可能处理其他 DP 的图像 |
| `runtime.py::EncoderParallelPlan.encode_and_route` | feature exchange 只在 TP 或 CP×TP 范围内 | 生产者和 decoder owner 可能分属不同 DP |
| `planning.py::build_destinations` | 所有 decoder layout 被要求有相同的全局序列长度 | 不同 DP 样本可有不同长度；只应在各自 DP 内比较 |
| `exchange.py::route_features` | `item_id` 是单个样本内的序号，通信组不超过 CP×TP | 不同 DP 都有 `item_id=0`；全域目的 rank 需要唯一键 |
| `models/deepseek_v41/adapter/distributed/encoder_parallel.py::DeepseekV41VisionBridge._pack_local_inputs` | 从本地完整 patch tensor 按 offset 切片 | 需要接受已路由到本 rank 的 patch shard |

这些文件在仓库根目录下分别位于 `hyper_parallel/`。修复前的
`hyper_parallel/data/batching/get_batch.py::OmniParallelBatch.__call__` 和
`hyper_parallel/data/parallel/batch_parallel.py::CPBatchSharder._shard_configured_fields`
已经让同一 DP 样本的 CP/TP rank 保留完整 modality 字段。全局路径可从每个 DP 的
`(cp=0,tp=0)` rank 读取一次完整 patch tensor，不需要改变 DataLoader 的 DP 采样语义。

### 2.1 修复前图像数据怎样到达每张卡

以 `DP=2、CP=2、TP=2` 为例，DP0 的样本 A 和 DP1 的样本 B 由 DP 采样器分别
选出；同一 DP 内的各 CP 坐标使用相同的 DP 采样坐标。`DataLoaderParallelContext`
在每个 CP 坐标选择 `TP=0` 读取 DataLoader，所以 DP0 的 `(CP0,TP0)` 与
`(CP1,TP0)` 都会取得 A。接下来有两个性质不同的操作：

| 阶段 | token 字段 | 图像字段 `pixel_values` 与 `image_*` |
| --- | --- | --- |
| `CPBatchSharder._shard_configured_fields` | `input_ids`、`labels`、`token_types` 等 `OMNI_CP_TOKEN_FIELDS` 按 CP 切片 | 不在 CP token 字段集合中，保持原样；每个 CP 源 rank 拿到整份图像 patch |
| `TPBatchBroadcaster._broadcast_mapping` | CP-local token 搬到设备并广播给同 CP 的 TP rank | 整份图像 patch 也搬到设备并广播给同 CP 的 TP rank |

因此，在这个八卡例子里，rank `0..3` **各有一份 A 的完整设备端**
`pixel_values`，rank `4..7` **各有一份 B 的完整设备端** `pixel_values`。
这是跨设备的四份副本；不能把它误算成单张卡上同时有四份。CP 缩短的是 token
字段，不是图像 patch 数；TP 广播也没有分片图像。修复前的
`DeepseekV41VisionBridge._pack_local_inputs` 先从本地完整 `pixel_values`
按 offset 取图，再取本 ECP rank 的 patch shard，所以旧实现需要这些副本。

### 2.2 修复前连续多次 `get_batch` 与 OOM 的关系

以下描述修复前的预取路径。另一层放大发生在**同一张卡的时间轴**上。
`VLMTrainer._build_collate_fn` 用
`calculate_num_micro_batches` 算出
`num_micro_batches = global_batch_size / (micro_batch_size × dp_size)`；
`VLMTrainer.train_step` 随后连续调用这么多次 `get_batch`，把所有返回值放入
`training_batches`，直到全部取完才进入第一轮 forward/backward。每次调用中，
`TPBatchBroadcaster._move_mapping_to_device` / `_broadcast_mapping` 已把 batch tensor
放上设备；`model_inputs` 与 `loss_inputs` 只是共享这些 tensor 的两个字典，
不应把它们误算成两份数据。循环中虽在每次 backward 后执行
`training_batches[micro_step] = None`，这无法降低**第一轮 forward 之前**
所有 batch 同时在场的峰值；`first_training_batch` 这个局部变量还会引用第一份
batch 直到 `train_step` 返回。

若现场配置确实得到 `num_micro_batches=6`，且快照显示同一 rank 的每次
`get_batch` 增加约 8.9 GB **仍然存活的设备张量**，那么进入第一轮 forward 前
仅这六份 batch 就约占 `6 × 8.9 = 53.4 GB`，还未计模型参数、优化器、通信缓冲
和后续激活。这个时间模式与“六次 `get_batch` 后 OOM”高度吻合，但单凭函数名
不能认定 8.9 GB 全是图像：1M 个 `int64` token 约 8 MB，仍须按字段和分配栈
核实具体大 tensor。
若这批次没有 `pixel_values`，图像字段复制就不是该次分配的原因；仍需检查其他
大字段或运行时分配，但多 microbatch 预取的存活时间问题依然存在。

核实顺序：先用实际配置算出 `num_micro_batches`，再看快照中六次分配是否发生在
**同一个 optimizer step、同一个 rank、第一次 forward 之前**；将分配栈定位到
`_move_mapping_to_device` 的 `.to(self.device)`、`_broadcast_mapping` 接收端的
`torch.empty`，或其他调用点；记录每个大字段的 `shape × element_size`，特别是
`pixel_values`，并对比每次 `get_batch` 后的 `memory_allocated` 与
`memory_reserved`。前者对应活张量，后者还包含 allocator 缓存；只看 reserved
的阶梯形增长不足以证明六份数据仍同时存活。

这也改变了本方案的显存边界：**只把 ECP 扩展到跨 DP，不会自动消除上述两层
放大。**当前 `sample` 和 `global` 模式都让 `OmniParallelBatch` / `TPBatchBroadcaster` 仅在每个
DP 的 canonical patch 源 rank 保留完整设备端 `pixel_values`，其他 rank 只获得图像元数据
与路由后的有效 patch。`preflight_batch` 已允许非源 rank 缺少完整像素。
`VLMTrainer` 先在 CPU 侧预读本 step 的 CP-local batch，精确统计有效 token 总数，
再逐个创建设备 batch、完成 forward/backward 并释放设备输入。CPU 侧的 CP 切片
只做一次；因此设备端不再预留六份 batch，但主机内存仍可能持有六份原始图像，
1M 序列验收还需同时记录 CPU 峰值与设备峰值。

### 2.3 为什么不能直接用 `encoder_dp/encoder_cp` 替换 OmniBatch 的 `dp/cp`

`encoder_dp` 不是独立的数据采样维度，而是 ViT 工作桶编号。`sample` 模式在
**一个 decoder DP 样本域内**把 `CP×TP` 重排为该工作桶和 ECP；`global` 模式
在 `DP×CP×TP` 全域重排，但仍不改变 DataLoader 的样本所有者。
`assign_lpt` 负责把这个样本的图像任务分给工作桶；它不读取新样本。

以第 5 节的八卡配置为反例：decoder `DP=2、CP=2、TP=2`，所以同一 microstep
有样本 A、B 两个 decoder 样本域。全域 `ECP=8` 后只有
`encoder_bucket_count=W/ECP=1`。若直接让 OmniBatch 以
`encoder_dp=1` 代替 decoder DP 采样，现有
`global_batch_size/(micro_batch_size×dp_size)` 会按 1 而非 2 计算 microstep 数；
单靠 ViT 工作桶编号也无法表达“A 归 decoder DP0、B 归 decoder DP1”这两个目的地。
若再让 OmniBatch 以 `encoder_cp=8` 代替 decoder CP 切 `input_ids`、`labels`，
token 会变成八份，而 decoder 的 CP/SP 布局仍要求每个样本按 CP2、TP2 定位。
直接替换坐标会同时改变采样数量、文本 token 归属和 loss 归一化。

更完整的数据层重构应让一个 microbatch 同时携带两种布局，而不是让一个
`dp/cp` 字段兼任两种含义：

| 职责 | 使用的坐标 | 数据去向 |
| --- | --- | --- |
| 样本身份与 loss 归属 | decoder DP，或显式的逻辑 `sample_owner` | DataLoader 为每个 decoder DP 选择样本；同一样本的 loss 最终回到该 DP |
| 文本切片 | decoder CP/TP/SP | `input_ids`、`labels`、`token_types` 到原 decoder token owner |
| 图像任务调度 | 全域 `encoder_bucket/encoder_cp` | 完整 patch 仅由 canonical 源持有，按任务键分发有效 shard |
| 图像结果归还 | `(sample_owner, item_id)` 与 decoder layout | feature 只到所属样本的 decoder token owner |

因此应把 OmniBatch 演进为“同一逻辑样本，分别构造两个模型输入”的协调者：
首版保留 decoder DP 的样本采样和 token sharding，另按 encoder topology 构造
ViT 的任务表与 patch shard。长期若希望按 encoder bucket 读取源数据，也必须增加
独立的样本到 decoder DP 映射及文本路由，不能只改传给 OmniBatch 的两个
size/rank 参数。这个选择本身也不解决第 2.2 节的多 microbatch 设备预取。

## 3. 拓扑、任务键与数据契约

### 3.1 两种工作域

在 `hyper_parallel/distributed/encoder_parallel/config.py::EncoderParallelConfig` 增加
`group_scope: Literal["sample", "global"] = "sample"`，并在 `__post_init__` 验证。
`sample` 路径维持当前 `cp×tp` 域。`global` 路径使用一个全域 encoder arena：

```text
P = cp_size × tp_size
W = dp_size × P                    # pp_size=1
arena_rank = dp_rank × P + cp_rank × tp_size + tp_rank
encoder_bucket = arena_rank // encoder_cp_size
encoder_cp_rank = arena_rank % encoder_cp_size
encoder_bucket_count = W // encoder_cp_size
```

`arena_rank` 是 mesh 的局部线性序号，不是任意环境下的 `dist.get_rank()`。进程组
成员与源 rank 由 `device_mesh.rank_list` / 派生子 mesh 给出，不能假设全局 rank
总是 `0..W-1`。全局 arena 可以选 `device_mesh.flatten("encoder_global")`，再
`_unflatten` 为 `(encoder_bucket_count, encoder_cp_size)`。样本域继续保留原来的
`("cp", "tp")` 派生网格。两条路径使用不同的 flatten 名称，避免 mesh 缓存重名。

### 3.2 唯一任务键

在 `contracts.py::EncoderWorkItem` 末尾增加默认值为 `0` 的
`owner_dp_rank: int`，提供 `key -> tuple[int,int]` 属性，返回
`(owner_dp_rank, item_id)`。字段放末尾可保留现有位置参数测试。DeepSeek 的
`build_deepseek_v41_work_items()` 仍生成样本内 `item_id`；
`EncoderParallelRuntime.preflight_batch()` 在解析后用 `dataclasses.replace` 标记
当前 `dp_rank`。全域分桶、patch 描述符、feature 映射和目的地映射一律使用 `key`。

`owner_dp_rank` 表示 **decoder 样本所有者**，与任务被分配到哪个 encoder bucket
无关。每个样本保留自己的 `batch_index`、`decoder_token_start`、patch offset 和
`metadata`；offset 只在所属 DP 样本的 `pixel_values` 中有意义。

### 3.3 元数据和 payload（后续优化目标）

每个 DP 的 `(cp=0,tp=0)` rank 是该样本的 patch 源。目标是先在原 `cp×tp` parent
中校验 `cp_local_length` 和图像任务描述相同，再由这些源 rank 交换以下小型元数据：

```text
(owner_dp_rank, cp_local_length, [EncoderWorkItem...],
 pixel_dtype, pixel_shape_tail)
```

`pixel_shape_tail` 对 DeepSeek 是 `(3, patch_size, patch_size)`。所有活跃样本的
patch dtype 和尾部 shape 必须一致；没有图片的 DP 也要参与元数据通信，并从
全域元数据构造零长度发送缓冲。每个 rank 由相同的元数据决定分桶和通信 split，
无需传输 `input_ids`、`labels` 或 decoder hidden。

### 3.4 OmniBatch 的双模型输入契约

本节是后续接口目标。当前实现复用扁平 `model_inputs`，用
`encoder_parallel_plan` 携带已路由的本地 patch，详见 §8。

全局模式下，OmniBatch 先从一个逻辑样本产生稳定的样本键，再生成两个按不同
mesh 放置的输入。`decoder_inputs` 含 CP/TP-local 的 `input_ids`、`labels`、
`token_types`、position/packed metadata 和 loss 字段；不携带完整设备端
`pixel_values`。`encoder_inputs` 含全域计划与本 ECP rank 的**有效 patch shard**；
即使本 decoder DP 没有图片，只要全域有图，该 rank 仍可能持有别的样本的
patch shard。二者只通过 `(owner_dp_rank, item_id)` 和 feature route 相连。

```python
@dataclass(frozen=True)
class EncoderBatchInputs:
    plan: EncoderParallelPlan
    local_patches: torch.Tensor  # 按 local_segments 顺序，仅有效行

# 维持 Trainer 现有的二元返回形状；global 模式的 model_inputs 是扁平 decoder
# 字段加一个明确命名的 encoder 输入对象，而不是混装完整 pixel_values。
model_inputs = {**decoder_inputs, "encoder_inputs": encoder_inputs}
return model_inputs, loss_inputs
```

构造顺序是“读取逻辑样本 → decoder token 切片与必要元数据广播 → 汇总全域图像任务
并建立计划 → canonical 源搬运完整 patch → 按 ECP 路由有效 patch → 返回两份输入”。
`DeepseekV41Model.forward` 在全局模式把 `encoder_inputs.local_patches` 交给 ViT，
再把 feature route 的结果并入本地 decoder hidden。当前 `sample` 与 `global`
模式都保留扁平 token 字段和 `encoder_parallel_plan` 接口，均不把完整
`pixel_values` 返回给模型；两条模式的
输出记录可继续共用 `_merge_image_embeddings`。路由完成后应尽早释放源 rank 的
完整设备端 patch 引用，但本地 shard 要活到该 microbatch 的 backward 完成。

## 4. 按函数实施

下表给出完整双输入接口与分层元数据通信的**目标改造清单**，不是对当前代码
逐行行为的断言。为了尽量复用旧接口，首轮只落地其中必要的拓扑、路由和显存
修改；实际函数、已实现行为与剩余工作列在 §8。

| 顺序 | 修改函数 | 明确的接口与行为 |
| --- | --- | --- |
| 1 | `config.py::EncoderParallelConfig.__post_init__`、`from_mapping` | 接受 `group_scope` 并拒绝其他值。配置对象只检查字段类型与取值；拓扑整除由 runtime 检查。 |
| 2 | `mesh.py::derive_reshaped_submesh` | 保留重排算法，允许从 `("dp","cp","tp")` 派生全域网格；建议增加显式 `parent_name` 参数，避免当前固定的 `encoder_parent` flatten 名在两种 scope 下冲突。单测检查 rank 顺序、组成员与本地 ECP 坐标。 |
| 3 | `runtime.py::EncoderParallelRuntime.__init__` | 先校验 PP、scope 整除（`sample` 用 `P % ECP == 0`，`global` 用 `W % ECP == 0`）和 `validate_ulysses_head_divisibility`，再创建 arena、`encoder_cp_mesh`、原 decoder parent mesh、DP leader mesh 与交换 mesh。错误信息给出 scope、域大小与 ECP。保存 `parent_rank`（样本内）和 `arena_rank`（全域）两个坐标。`sample` 分支保持现有组选择。 |
| 4 | `runtime.py::EncoderParallelRuntime.preflight_batch`、`batch_input.py::EncoderParallelInputAdapter.build_runtime_inputs`、`models/deepseek_v41/adapter/distributed/encoder_parallel.py::build_deepseek_v41_work_items` | 全局模式将图像任务元数据解析与完整像素校验拆开：所有 rank 校验 decoder layout 与轻量图像描述符，仅 canonical 源校验完整 `pixel_values`。把 preflight 与 `inner.build_runtime_inputs` 分成独立阶段，每阶段捕获本地错误并做全域状态归约，失败时所有 rank 一致退出。样本域模式保留旧校验。 |
| 5 | `runtime.py::EncoderParallelRuntime.finalize_plan` | `sample` 分支保持旧逻辑。`global` 分支先做 parent 内 digest 一致性检查，再由 DP leader 在其 `dp` 子 mesh 汇总各样本的工作项和 `cp_local_length`，并广播给本 parent。按 `(owner_dp_rank,item_id)` 稳定排序后，全 rank 得到同一任务表。若全域无图像，则所有 rank 一致返回 inactive plan。 |
| 6 | `planning.py::assign_lpt`、`build_segments` | `assign_lpt` 对全域任务按 `(-sequence_length²,owner_dp_rank,batch_index,item_id)` 确定性排序，先分配长图，生成 `W/ECP` 个 bucket；相同图像只分配一次。`build_segments` 仍按 bucket 内每张图切出 ECP 个 padded shard；空 bucket 仍生成 dummy segment。样本域的 tie-break 结果应保持不变。 |
| 7 | `planning.py::build_destinations` | 接受全域 `layouts` 和 rank 所属 DP 信息。对每个 item 只检查 `owner_dp_rank` 对应的 CP/TP rank，并仅在该 DP 内校验 `global_sequence_length`。返回 `dict[item.key, tuple[arena_rank,...]]`；无 owner、越界 image span 都在通信前失败。 |
| 8 | 新增 `distributed/encoder_parallel/input_exchange.py::build_patch_exchange_plan`、`route_patch_shards` | 从每个样本的 canonical source 把有效 patch 行送至分配到该图的 ECP rank。先按确定性描述符计算每个目的 rank 的 row split，再用全域 `dist.all_to_all_single` 接收、按 `item.key` 重排成 `local_segments` 顺序。空发送/接收仍参加 collective。对 shape、dtype、patch 总数和 `requires_grad` 做全域一致的预检查。 |
| 9 | `models/deepseek_v41/adapter/distributed/encoder_parallel.py::DeepseekV41VisionBridge._pack_local_inputs`、`encode_local` | 两种 scope 均接收已经排序的有效 patch shard；按 `EncoderSegment.local_valid_length` 消耗输入，补齐 dummy/padding，并根据 item 的 `vit_hw` 生成该 shard 的 RoPE。两种 scope 的像素输入梯度均受 source-only eager transport 限制。ViT block 按四层一组 checkpoint，反向恢复 attention context。 |
| 10 | `runtime.py::EncoderParallelPlan._execute_local`、`_gather_features` | 两种 scope 均消费 OmniBatch 已路由的有效 patch shard，向 bridge 传入按 segment 排序的有效 patch，再用现有 packed ViT/Ulysses 和可微 all-gather。所有 rank 在全域 active 时调用一次 ViT；`k_fsdp_max=max(1,max(len(bucket)))` 从全域计划确定，所有 rank 调用相同次数的 aligner/dummy，保持 FSDP 执行次序。 |
| 11 | `exchange.py::select_feature_exchange`、`route_features`、`_route_variable_a2a`、`unpack_records` | 增加 `GLOBAL_VARIABLE_A2A` 模式；全局模式总选 arena group 的可微 variable all-to-all。生产者是每个 bucket 的 ECP rank 0；源/目的记录用 `item.key`。发送目标仅为 `owner_dp_rank` 的 decoder rank。接收后检查每条记录的 owner 等于本 rank 的 DP，再按样本内顺序排列。 |
| 12 | `runtime.py::EncoderParallelPlan.encode_and_route` | 全局模式接受 `EncoderBatchInputs.local_patches`，以 `arena_rank` 和全域交换 mesh 调用 `route_features`；样本域沿用 TP gather/A2A 或 parent A2A。无本地 decoder 图片但全域 active 的 rank 仍执行 ViT、feature A2A，并返回零值梯度 anchor。 |
| 13 | `data/batching/get_batch.py::OmniParallelBatch.__call__`、`data/parallel/batch_parallel.py::TPBatchBroadcaster._broadcast_mapping`、`batch_input.py::EncoderParallelInputAdapter.build_runtime_inputs`、`trainer/vlm_trainer.py::_build_get_batch` | 沿用原 DP sampler；两种 scope 的 token/元数据按 decoder 布局广播，完整 `pixel_values` 只上 canonical 源设备；adapter 在 `get_batch` 内调用 `route_patch_shards`。§3.4 的显式双输入对象及配置诊断展示仍是后续工作。 |
| 14 | `models/deepseek_v41/modeling_deepseek_v41.py::DeepseekV41Model.forward`、`_prepare_image_embeddings` | 全局模式新增 `encoder_inputs` 参数，把其本地 shard 交给 `encode_and_route`；decoder 侧仍按本地 `token_types` 合并归属自己的记录。样本域模式保留原扁平图像参数接口。 |
| 15 | `trainer/vlm_trainer.py::train_step`、`_forward_backward_micro_batches` | 每次只持有当前 microbatch 的设备输入，完成该次 forward/backward 后释放引用，再取下一份；同步调整 step/microstep 回调与 token 计数，保证梯度累积和 loss 语义与旧实现相同。 |

表中 `distributed/...`、`models/...`、`trainer/...` 均以 `hyper_parallel/` 为根。
`DeepseekV41Model._prepare_image_embeddings` 在全局模式从
`encoder_inputs.plan.encode_and_route(encoder_inputs.local_patches)` 得到本地记录，
在样本域模式沿用旧调用；两者继续共用图像 token merge，并增加断言防止其他 DP
的 record 进入本 decoder。

### 4.1 接口草案与数据顺序

下列签名固定跨模块的输入输出契约，具体数据类放在 `input_exchange.py`，避免
`input_exchange` 反向导入 runtime：

```python
WorkKey = tuple[int, int]  # (owner_dp_rank, item_id)

@dataclass(frozen=True)
class PatchExchangePlan:
    input_splits: tuple[int, ...]          # 每个目的 arena rank 的 patch 行数
    output_splits: tuple[int, ...]         # 每个源 arena rank 的 patch 行数
    send_slices: tuple[tuple[WorkKey, int, int, int], ...]
    receive_order: tuple[tuple[WorkKey, int], ...]
    patch_dtype: torch.dtype
    patch_shape_tail: tuple[int, int, int]

def build_patch_exchange_plan(
    *, buckets: tuple[tuple[EncoderWorkItem, ...], ...],
    encoder_cp_size: int, decoder_parent_size: int,
    arena_rank: int, patch_dtype: torch.dtype,
    patch_shape_tail: tuple[int, int, int],
) -> PatchExchangePlan: ...

def route_patch_shards(
    *, pixel_values: torch.Tensor | None,
    plan: PatchExchangePlan, arena_mesh: DeviceMesh,
) -> torch.Tensor: ...

def encode_local(
    self, packed_patches: torch.Tensor,
    segments: Sequence[EncoderSegment],
) -> torch.Tensor: ...
```

`send_slices` 的元组为 `(item.key, destination_arena_rank, local_start,
local_valid_length)`；具体切片起点是该 item 在所属 DP 的
`pixel_values` 中的 `payload_start + local_start`。`receive_order` 记录从各源
rank 收到的 `(item.key, valid_length)`，用于恢复 `local_segments` 的顺序。
所有 split 都以 **patch 行**计数，`route_patch_shards` 返回只含有效行、按
`local_segments` 顺序拼接的 `[sum(local_valid_length), 3, P, P]` tensor；padding
和 dummy 在 `encode_local` 内构造。零有效行的 segment 仍占据任务表位置。
`EncoderParallelPlan.destinations` 相应改为
`dict[WorkKey, tuple[int, ...]]`，其中 rank 是 arena 局部序号；`sample` 路径
虽沿用原通信模式，也统一使用该键，并更新内部 UT 的预期映射。

### 4.2 元数据 collective 的目标顺序（尚未实现）

全局模式的 `finalize_plan()` 后续可按以下顺序优化，所有 rank 必须遵守同一顺序。
当前 `_finalize_global_plan` 使用全域 `all_gather_object` 汇总各 rank 元数据，再逐 DP
校验副本；`route_patch_shards` 使用一次全域 `all_gather_object` 校验 patch 源。
它没有独立的 parent digest、leader 子组交换和 parent broadcast。

1. 所有 rank 本地运行 `preflight_batch`，捕获异常并做第一次全域错误标记归约；
   全部通过后才运行 `inner.build_runtime_inputs`，然后做第二次全域错误标记
   归约。失败 rank 的诊断应随统一状态传播；不能让成功 rank 先进入下一步。
2. 在每个原 decoder parent 内 `all_gather` metadata digest，校验 CP/TP 副本一致；
   随后做一次全域错误标记归约。任何 parent 不一致时，所有 rank 都在 leader
   元数据交换前退出，不能只有出错 parent 抛异常。
3. 只有 `(cp=0,tp=0)` 的 DP leader 在 `device_mesh["dp"]` 对应的 leader
   group 执行 `all_gather_object`，交换小型 Python 元数据；其余 rank 不进入该组。
4. 每个 leader 在本 parent `broadcast_object_list` 全域任务表及各 DP 的
   `cp_local_length`；本 parent 的所有 rank 都参与该广播。
5. 所有 rank 本地构建相同分桶、目的地、patch split 与 feature split。对 payload
   校验状态再做一次全域错误标记归约，然后进入 patch all-to-all。

首版用 object collective 仅传小型元数据；patch 和 feature 都使用 tensor
collective。之后可把元数据编码为定长/变长 `int64` tensor，减少 HCCL object
collective 的开销。组构建须在所有 rank 的 runtime 初始化阶段按统一顺序发生。
各错误同步点共用一个 `_agree_or_raise(phase, local_error, arena_group)` 辅助函数：
先 `all_reduce(MAX)` 失败标记；若任一 rank 失败，所有 rank 再以固定顺序汇集
`(arena_rank, 简短错误文本)`，选择最小失败 rank 的记录并抛出同一条异常。
正常路径不做错误文本的 object collective。该函数用于 preflight、inner adapter、
parent digest 和 payload 校验，不能只在出错 rank 本地 `raise`。

### 4.3 patch 与 feature 的反向语义

`route_patch_shards` 的输入是训练数据，不是模型参数；首版使用 eager
`dist.all_to_all_single` 并拒绝 `pixel_values.requires_grad=True`。每个 rank 的
ViT 对接收的 patch 创建正常参数计算图。aligner 只在 bucket 的 ECP rank 0
产生真实 feature；其他 rank 的 dummy aligner 和零值 anchor 保持 FSDP 模块调用
次数一致。`GLOBAL_VARIABLE_A2A` 继续使用现有
`differentiable_all_to_all_single`，使每个 decoder owner 的 feature 梯度返回
生产者；`_gather_features` 的可微 all-gather 再把梯度送回所有 ViT shard。

全局模式 active 的判断基于全域任务表，不能只检查当前 DP 的 `items`。即使某个
DP 样本没有图像，它的 rank 仍可能替其他 DP 编码，并必须通过零值 anchor 进入
反向。对所有 rank 保持固定的 ViT、aligner、patch A2A、feature A2A 调用顺序，
是避免 FSDP/collective 死锁的前提。梯度缩放仍由现有 Trainer/FSDP 负责；不能
因为 ViT 计算迁移到别的 DP rank 就再次除以 DP 或 ECP。验收时要比较完整参数
梯度与单卡参考，并检查不同 DP 负载下的缩放。

## 5. 端到端示例：ECP8 跨越两个 DP 样本

以下用 8 个 rank 的心智模型说明目标端到端路径。现有 CPU/Gloo worker 只验证
图像 feature 路由和可微反向；真实 token merge、token-weighted loss 和标记位
零梯度仍需补测。训练配置中对应的片段是：

```yaml
accelerator:
  tp_size: 2
  cp_size: 2
  pp_size: 1
  sequence_parallel: true
  encoder_parallel:
    enabled: true
    group_scope: global
    encoder_cp_size: 8
fsdp_config:
  dp_shard_size: 4
training:
  micro_batch_size: 1
  global_batch_size: 2
```

在 8 rank 下，`dp_size=2`，`W=2×2×2=8`。合成 ViT 设置 hidden size 32、
8 个 Q/K/V head（每 head 4 维）、patch size 2 和 aligner downsample ratio 2，
满足 Ulysses 的 `8 % 8 == 0` 和 ViT
RoPE 的维度要求。本例用于验证新链路；把同一配置用于真实
DeepSeek checkpoint 前，需先检查该模型 **实际** ViT head 数是否能被 8 整除。
`global_batch_size=2` 让两个 DP 各拥有一个样本；现有 E0 配方的 batch 和 ViT
head 设置不能直接套用这个片段。

| arena rank | DP | decoder CP | decoder TP | ECP rank | 本地 decoder token 区间 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0 | 0 | 0 | 0 | `[0,4)` |
| 1 | 0 | 0 | 1 | 1 | `[4,8)` |
| 2 | 0 | 1 | 0 | 2 | `[8,12)` |
| 3 | 0 | 1 | 1 | 3 | `[12,16)` |
| 4 | 1 | 0 | 0 | 4 | `[0,4)` |
| 5 | 1 | 0 | 1 | 5 | `[4,8)` |
| 6 | 1 | 1 | 0 | 6 | `[8,12)` |
| 7 | 1 | 1 | 1 | 7 | `[12,16)` |

两个 DP 样本都取 `cp_local_length=8`，启用 decoder SP，所以每个 TP rank
负责 4 个 token。DP0 的图片 A 使用 4×4 ViT patch 网格，共 16 patch；
aligner 的 2×2 LLM 网格输出 4 行 feature，图片 token span 为 `[2,10)`。
DP1 的图片 B 使用 2×4 ViT 网格，共 8 patch；1×2 LLM 网格输出 2 行 feature，
图片 token span 为 `[4,9)`。两张图的样本内 `item_id` 都是 0，但全域 key 分别
是 `(0,0)` 和 `(1,0)`。DP0 的完整 A patch tensor 位于源 rank 0，DP1 的完整
B patch tensor 位于源 rank 4；同 DP 其他 CP/TP 副本不会重复发送。

合成 worker 给两个 DP 各构造一份长度为 16 的原始 token 序列，随后由
`OmniParallelBatch` 切成 CP-local 长度 8。图片字段可按下表构造；`token_types`
在图片区间之外使用普通文本类型，在区间内按模型要求填写：

| DP | `pixel_values.shape` | `image_patch_offsets` | `image_vit_grid_hw` | `image_llm_grid_hw` | `image_batch_indices` | `image_token_starts` | 图片区间的 `token_types` |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | `[16,3,2,2]` | `[0,16]` | `[[4,4]]` | `[[2,2]]` | `[0]` | `[2]` | `[IMAGE_START, IMAGE, IMAGE, IMAGE_NEW_LINE, IMAGE, IMAGE, IMAGE_NEW_LINE, IMAGE_END]` |
| 1 | `[8,3,2,2]` | `[0,8]` | `[[2,4]]` | `[[1,2]]` | `[0]` | `[4]` | `[IMAGE_START, IMAGE, IMAGE, IMAGE_NEW_LINE, IMAGE_END]` |

图片区间之外的 `token_types` 填 `TEXT=-1`；`TEXT` 和 `IMAGE_*` 常量来自
`hyper_parallel/models/deepseek_v41/adapter/data/image_processor.py`。
`input_ids`、`labels` 均为 `[1,16]`。完整端到端测试应另外构造真实 loss mask
和有效文本 token；现有 CPU/Gloo worker 的 `token_types` 与 `labels` 为零值，
只对收到的每条 feature 求和反向。图片 token 的局部切片按 decoder 布局完成；
完整 patch 只在 canonical 源上设备并用于全局路由，其他 rank 的
`encoder_inputs.local_patches` 只含本 ECP rank 的有效行。

1. **读入与规划。** `OmniParallelBatch.__call__` 继续让两个 DP 读各自样本；
   `preflight_batch` 得到各自的 item。当前所有 arena rank 交换元数据并逐 DP
   校验副本。全域 `assign_lpt` 得到一个 bucket，含 A 和 B。
2. **patch 分发。** `build_segments` 对 A 切成 8 份，每 rank 2 patch；对 B
   切成 8 份，每 rank 1 patch。`route_patch_shards` 让 rank 0 将 A 的对应 2 行
   送到每个 ECP rank，rank 4 将 B 的对应 1 行送到每个 ECP rank。每个 rank
   最终按 `(A 的 2 行, B 的 1 行)` 打包，执行一次 ViT；attention 按 A/B
   segment 分别做 Ulysses，不能让两张图互相注意。
3. **投影与回传。** 8 个 rank 通过可微 all-gather 得到各自 segment 的完整
   hidden；只有 ECP rank 0，即 arena rank 0，运行真实 aligner，得到 A 的
   4 行和 B 的 2 行。目的地规划只看所属 DP：A 送到 rank `0,1,2`，B 送到
   rank `5,6`。rank `3,4,7` 接收零行，但仍参加全域 feature A2A。
4. **decoder 与反向（待补真实 merge 测试）。** 每个 rank 只 merge 属于本 DP 且与本地 token 区间
   相交的记录。A 的 4 行 feature 实际在 rank 0 使用第 0 行、rank 1 使用
   第 1～3 行；rank 2 只碰到图片的换行和结束标记，所以会收到 A 的记录，
   但其 feature 梯度为零。B 的 2 行都在 rank 5 使用；rank 6 只碰到结束标记，
   对 B 的 feature 梯度也为零。`loss.backward()` 后，各 owner 的有效梯度及
   零梯度经可微 A2A 返回 rank 0；随后可微 all-gather 的反向将 ViT 梯度分配给
   8 个 ECP rank。rank `3,4,7` 通过零值 anchor 保持反向图和 FSDP 调用对齐。

现有 worker 使用 `feature.sum()`，因此 rank 2、rank 6 收到记录时也贡献非零
feature 梯度；参考端相应令 A 的梯度乘 3、B 乘 2。上段的零梯度行为是
`_merge_image_embeddings` 接入真实图片标记后应补的断言，不能用当前 ST 结果替代。

完整目标接口的调用链（当前实现仍使用扁平输入，实际链路见 §8）：

```text
VLMTrainer._build_get_batch
  -> OmniParallelBatch.__call__
    -> decoder token sharding / metadata broadcast
    -> EncoderParallelInputAdapter.build_encoder_inputs
      -> EncoderParallelRuntime.preflight_batch / finalize_plan
      -> route_patch_shards -> EncoderBatchInputs(local_patches, plan)
  -> DeepseekV41Model.forward(encoder_inputs=...)
    -> _prepare_image_embeddings -> EncoderParallelPlan.encode_and_route
      -> DeepseekV41VisionBridge.encode_local
      -> EncoderAttentionContext / Ulysses -> _gather_features
      -> DeepseekV41VisionBridge.project -> route_features(global A2A)
    -> _merge_image_embeddings -> decoder layers -> loss.backward()
```

当前 `tests/torch/encoder_parallel/test_encoder_parallel.py` 已包含
`test_global_runtime_dp2_cp2_tp2_ecp8`，由薄 launcher 用 `torchrun_case`
启动 8-rank worker。feature 路由/反向复现命令为：

```bash
pytest -vs tests/torch/encoder_parallel/test_encoder_parallel.py::test_global_runtime_dp2_cp2_tp2_ecp8
```

## 6. 测试与落地顺序

| 阶段 | 测试位置与具体断言 |
| --- | --- |
| 拓扑与计划 UT | 扩展 `tests/ut/distributed/encoder_parallel/test_runtime.py`、`test_planning.py`：`dp2/cp2/tp2/ECP8` 可构建；ECP 不整除 W、head 不整除 ECP 均在组派生前失败；同一 `item_id` 的两个 DP 任务不相撞；不同 DP 的 decoder 长度可不同；只路由给所属 DP；置换 `rank_list` 后仍使用 mesh 局部序号。 |
| 双模型输入 UT | 扩展 `tests/ut/data/vlm/test_get_batch_cp.py` 的 Omni batch 测试：全局模式的 decoder 输入按原 CP/TP 切片，目标 `encoder_inputs.local_patches` 按 ECP 段排列；非 canonical rank 不把完整 `pixel_values` 搬到设备；文本-only decoder DP 仍能收到其他样本的 ViT shard。当前样本域模式保持扁平 token 字段和 `encoder_parallel_plan`，但已移除返回给模型的完整 `pixel_values`。用两个不同样本验证不会按 `encoder_bucket_count` 改变采样数。 |
| patch/feature UT | 新增 `test_input_exchange.py`，mock 通信并验证源 rank、split、接收顺序、空图像 rank 与 dtype/shape 错误；加入 `patch_count < ECP` 和 patch 数不能整除 ECP 时的零有效行/padding 断言。扩展 exchange 测试，验证跨 DP 目的地及零行 rank 的反向 anchor。 |
| CPU/Gloo ST | 在 `tests/torch/encoder_parallel/` 添加薄 launcher 和 `_test_*.py` worker；运行本节 8-rank 示例，从真实 `OmniParallelBatch.__call__` 的双模型输出进入 `DeepseekV41Model._prepare_image_embeddings`，对 merge 后的图像 token 构造标量 loss 并反向。比对每 rank 的记录、实际 merge 的 feature 行、ViT/aligner 参数梯度与逐 DP 参考。再覆盖一个 DP 文本-only、两 DP 长度不同、空 bucket、ECP 小于 W、任一 rank preflight 失败、parent metadata 不一致、跨 DP patch dtype/shape 不同；错误用例要求所有进程在超时前报同一诊断而不死锁。launcher 遵守仓库规则，不导入 `torch` 或 `hyper_parallel`。 |
| microbatch 存活 UT | 在 `tests/ut/trainer/` 用可弱引用的大 batch 验证 `train_step` 在读取下一份设备 batch 前完成前一份 backward 并释放引用；比对多 microbatch 的梯度累积、step token/sample 计数、回调次数和 loss 归一化，防止用降低 global batch 的方式掩盖预取问题。 |
| NPU 训练验证 | 用真实 DeepSeek V4.1 crop 检查 ViT head 整除后，经 `VLMTrainer` 完成含不同 DP 图片的一个实际训练 step，检查 token-weighted loss 与参数梯度；记录显存与 step 时间，并与同样全局 batch、其余训练设置相同的 `group_scope=sample, encoder_cp_size=4` 合法基线比较。HCCL 专项用例断言主例的零行 feature 发送/接收，以及空 bucket 场景的零行 patch 接收均完成且反向无 hang。性能数据分别标记 ECP8/ECP4。 |

### 6.1 梯度数值参考与容差

NPU 对拍固定同一初始 checkpoint、样本 ID 与顺序、loss mask、随机种子和
optimizer step，分别运行 `sample/ECP4` 与 `global/ECP8`。在 optimizer 更新前
记录 ViT、aligner 的参数梯度；对 FSDP shard 按参数原顺序重组后比较同名张量，
同时核对实际输入 hash、全局有效 token 数与有限性。参考 loss 是两个 DP 样本
**所有有效目标 token** 的加权平均；`IGNORE_INDEX` 和 `loss_mask` 的交集遵循
`trainer/runtime/loss_aggregation.py::count_loss_token`。

实现中的每个 decoder rank 使用 `trainer/runtime/metrics.py::mean_global_loss`
计算反向项。对本例启用 SP 的布局，记 `L_r` 为本 rank 的局部 loss，`n_r` 为
TP 组归约后的有效 token 数，`N` 为固定 TP 坐标下 DP×CP 组汇总的 step token 数。
代码中的反向系数是：

```text
L_backward,r = L_r × n_r / N × (dp_size × cp_size) / tp_size
```

`dp_size × cp_size` 补偿 FSDP 的平均；`/ tp_size` 是 SP 路径现有缩放。
新 encoder 路由不能再对梯度乘或除以 DP、ECP。验收时先比 CPU/FP32
小模型的完整参数梯度（`atol=rtol=1e-4`，与现有 encoder ST 一致）；
完整 NPU/BF16 验收建议沿用
`examples/training_demo/deepseek_v41/deepseek_v41_validation.yaml` 的跨拓扑
预设：完整张量满足 `max_abs ≤ 1e-3` **或** `relative_l2 ≤ 2e-2`；
只保留摘要的大张量满足 `l2_norm_max_abs ≤ 2e-3` **或**
`l2_norm_relative ≤ 2e-2`。标量 loss 满足绝对误差 `≤ 5e-3` **或**
相对误差 `≤ 5e-3`；梯度范数满足绝对误差 `≤ 0.125` **或**
相对误差 `≤ 5e-3`。loss 与梯度范数各自都必须通过，另检查 shape
和非有限值。容差在运行前固定，不能按结果放宽。本次针对性十步矩阵的
运行前 ledger 对梯度范数使用 `relative ≤ 1e-2`，与本段建议的 `5e-3`
不同；两种判定及差异均在 §8 报告。

实现顺序建议为：先完成唯一键和全域拓扑的纯规划 UT；再接 metadata 与 patch
transport 的 CPU/Gloo ST；最后接可微 feature route 和真实 ViT 参数梯度 ST。
每个阶段都保持 `group_scope=sample` 的现有测试通过。修改 `.md` 后运行
`python3 .agent/scripts/check_agents_catalog.py`，并单独检查本文件与索引链接。

## 7. 验收标准

- `dp=2, cp=2, tp=2, ECP=8` 在全局模式完成一个含两张不同 DP 图片的
  forward/backward；旧样本域模式继续拒绝 `ECP=8`。
- 每张图只由一个 encoder bucket 编码一次，feature 只到所属 DP 的实际 token
  owner；不同 DP 的相同 `item_id` 不串线。
- 文本-only DP、空 bucket、变长 decoder 序列均不死锁；所有 rank 的 ViT、aligner
  与 A2A 调用顺序一致。
- ViT/aligner 参数梯度与按 Trainer 实际 loss 归一化对两个 DP 样本计算并聚合的
  参考一致；输入像素梯度在两种 scope 下按上述约束明确拒绝，不能静默丢失。

## 8. 首轮实施状态（2026-10-29）

已加入显式 `group_scope=global`，把 Encoder-CP 分组扩至 `DP×CP×TP`；沿用原任务规划、
ViT 执行和可微 feature exchange，实现跨 DP 的元数据汇总、源 rank 图像 patch 分发、
按原 decoder DP 归还 feature。`group_scope=sample` 仍为默认值。VLM Trainer 每次读取
并消费一个 microbatch，避免一次训练 step 预先持有全部设备 batch。

针对图像显存问题，`sample` 和 `global` 两种模式均只在每个有图样本的 CP0/TP0
把完整 `pixel_values` 搬上设备。`EncoderParallelInputAdapter.build_runtime_inputs` 在
`get_batch` 中完成 `route_patch_shards`，计划对象只保留本 encoder rank 的局部 patch；
`OmniParallelBatch.__call__` 在返回模型输入前移除完整 `pixel_values`。这样完整像素
张量不会被前向输入字典持有至反向结束。`DeepseekV41VisionBridge.encode_local` 在
训练时按四层一组调用 checkpoint，反向重算时重新进入原 Encoder Attention 上下文。

当前仍沿用扁平 `model_inputs` 与 decoder DP sampler；§3.4 和 §4 的显式双模型
输入对象是后续接口目标，encoder DP 尚未成为独立采样域。实际调用链是：

```text
VLMTrainer._build_get_batch
  -> OmniParallelBatch.__call__
    -> CPBatchSharder.shard
    -> TPBatchBroadcaster.broadcast(source_only_pixel_values=True)
    -> EncoderParallelInputAdapter.build_runtime_inputs
      -> EncoderParallelRuntime.preflight_batch / finalize_plan
      -> EncoderParallelPlan.prepare_patches -> route_patch_shards
    -> 移除 model_inputs.pixel_values，保留 encoder_parallel_plan
  -> DeepseekV41Model._prepare_image_embeddings
    -> EncoderParallelPlan.encode_and_route
      -> DeepseekV41VisionBridge.encode_local (每四层 checkpoint)
      -> _gather_features -> bridge.project -> route_features
    -> _merge_image_embeddings -> decoder -> loss.backward()
```

| 函数 | 当前行为 | 下一步可改进处 |
| --- | --- | --- |
| `OmniParallelBatch.prepare_step_sources` | 多 microbatch 时预读 CPU 上已切好的 CP-local batch，求真实 step token 总数；TP 组广播计数，预读错误全域同步 | 量测大图时的 CPU 峰值与预读耗时；若不足，再设计轻量 token 统计或可重放数据源 |
| `VLMTrainer._forward_backward_micro_batches` | 每次把一份预切 CPU batch 搬上设备，完成反向后释放设备引用；单 microbatch 直接流式执行 | 检查异步 DataLoader 预取与 host/pinned 内存寿命 |
| `EncoderParallelRuntime._finalize_global_plan` | arena 全体 rank 一次 `all_gather_object`，随后逐 DP 校验副本并规划 | §4.2 的 parent digest、leader 汇总与 parent 广播是减少对象通信的后续方案 |
| `EncoderParallelPlan.prepare_patches` / `route_patch_shards` | 两种 scope 都在 `get_batch` 里从各 DP 的 CP0/TP0 源发送有效 patch；全域源描述符用 `all_gather_object` 校验 | 量测对象通信与多图 all-to-all 的开销 |
| `DeepseekV41VisionBridge.encode_local` | 对 ViT block 每四层做 activation checkpoint，反向时恢复 Encoder Attention 上下文 | 真实 1M 工作负载下记录 ViT 激活与重算耗时 |
| `EncoderParallelPlan.encode_and_route` | 使用现有可微 feature exchange，把 `(owner_dp_rank,item_id)` 的记录送回 decoder owner | 补真实 token merge 与 token-weighted loss 的多 rank 数值测试 |

CPU/Gloo 8-rank 测试已覆盖 ECP8 跨 DP、ECP4 空桶与文本-only DP、`sample`
源 rank 独占像素及四层 ViT 的前向/参数梯度对拍；它用 `feature.sum()` 作为
反向标量，没有覆盖 §5 所述的真实 token merge。两卡 Gloo 测试覆盖 CPU
预读错误的全 rank 传播和变长批次的精确 token 总数。

**裁剪模型十步对拍。**本地 `output/training_demo/deepseek_v41/encoder_global_precision/oom_fix_matrix/`
保留运行 ledger、逐步 evidence、`execution.json` 和 `matrix-comparison.json`；
这些大文件不在源码压缩包内。两组分别从同一 warm checkpoint 续训第 2–11 步，
运行器均判定 `PASS`，每一步的全局输入 SHA256 和学习率相同。按运行前 ledger
的 BF16 标量阈值，比较器十步均 `PASS`：最大 loss 绝对差 `0.005743`，
最大更新前梯度范数绝对差 `0.257271`（相对差 `0.006444`）。若采用本节建议的
更严范数相对阈值 `0.005`，第 9、10 步未通过，不能写成该阈值下完全自洽。
这组 NPU 证据绑定 ledger 中的源码指纹
`be9c635b0c1d6eee305b68dacfda86348ac6f42dc4746b3c1af54f8dea2cfc06`；
本次 rebase 改变了数据管线依赖，旧结果不能证明新提交的 NPU 精度。

额外参数探针 `parameter-comparison.json` 的整体状态为 `FAIL`：
`main_grad_before_clip` 的 9,230 条记录中有 88 条超出
运行前参数阈值（`max_abs≤0.005` 或 `relative_l2≤0.005`），集中在 ViT 和图像
标记参数；`main_grad_after_clip` 的 8,370 条记录全部通过。更新后参数的
10,030 条记录跨布局数值差均在该阈值内，但其中 790 条因样本域基线自身的
`replicas_equal=False` 被严格探针判为失败；不能把这些判定等同于跨布局数值失败。
裁剪模型两种布局的设备峰值分配均约 `6.34 GiB`，不能据此推断真实 1M 负载的
显存节省。单次 CPU 合成测量（同目录的 `cpu_prefetch_benchmark.py` 和
`cpu-prefetch-benchmark.json`）使用六份各 1M token 的输入、CP2、四个 Torch
线程且不含图像；预读及创建 CPU 输入耗时 `0.0132 s`，单独计数耗时
`0.00185 s`，RSS 增长 `146.8 MiB`。这些数字只说明计数本身在该合成输入下
开销较小，不包含真实 DataLoader、图像解码和大图 patch；真实大图 CPU 峰值和
1M 序列 NPU 峰值仍需独立验收。
