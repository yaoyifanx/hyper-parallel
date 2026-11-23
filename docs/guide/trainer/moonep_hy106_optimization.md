# MoonEP 动态序列显存与 FP32 梯度累加优化

本次修改针对非融合与独立 megakernel；chunk、dedup 不在本轮优化和验证范围。VeOmni 的普通 EP + HyperParallel FSDP2 为对照。性能目标是降低 MoonEP 额外成本、接近普通 EP，同时保持参数、token 路由、有效序列和优化器更新语义。

首轮结果位于工作区 `glm/runs/hy106_comparison_20step/optimization_20261119`，第二轮为 `optimization_20261120_round2`。这是 16 rank、EP2、3 decoder layer 的减层验证，不能当作完整 106B 模型结果。

## 第二轮：删除重复存储，保持接近普通 EP

基于 `cda0586c`，三次独立 20 步无 profiler 重复，取第 11–20 步 world-max 墙钟均值；所有 rank 保持相同执行预取 depth=2、模型、数据与有效 token。以下是三次运行的均值和运行间标准差，显存为跨 rank、跨重复的最大值。

|实现|step time，s|相对普通 EP|11–20 allocated，GiB|1–20 allocated，GiB|
|---|---:|---:|---:|---:|
|VeOmni EP + HP FSDP2|2.404690 ± 0.023005|1.0000×|29.3055|29.3055|
|MoonEP 非融合，提前 RS 输入退休 + home view + W 工作区退还|2.547690 ± 0.033466|1.0595×|29.8425|29.9678|
|MoonEP megakernel，副本 dW 直写 G send|2.408240 ± 0.023415|1.0015×|30.9850|32.8777|

非融合相对本轮重新运行的冻结版本 35.6173 GiB，稳态 allocated 降低 5.7749 GiB；native 相对 32.2813 GiB 降低 1.2963 GiB。native 耗时与普通 EP 接近，未证明整步加速；NF 仍约慢 5.95%。全 rank 20 步 loss/norm 有限、无 skipped update、retry=0，不能用这些条件替代数值 oracle。native 第三次的 rank3 第 7 步仍有 32.8777 GiB 峰值，不能只报告稳态。

同输入 H4096/I1408、EP2/FSDP2、重算/no-sync、TopK2 的 direct-send 开/关对照，四 rank、两个角色的最终 FP32 梯度逐位一致。三次训练轨迹仍有漂移，普通 EP 自身 norm MAE 也达到约 0.225；本轮不宣称长期精度改善。

NF 推荐显式设置 `HP_FSDP_EARLY_RS_INPUT=1`、`HP_MOONEP_HOME_WEIGHT_VIEWS=1`、`HP_MOONEP_RELEASE_W_WORKSPACE=1`；native 设置 `HP_MOONEP_DIRECT_GRAD_SEND=1`，`HP_MOONEP_SEPARATE_TOKEN_STREAM=0`。两者继续使用下文共同 P/PF2 设置、真实 FP32 g、零常驻 RS 池与零额外 arena 保留。home/remote 两次 GMM 和 T 独立流已实测，本次未显示性能收益，均不纳入推荐。

详见 [第二轮报告](/home/ma-user/work/y00512198/gh_hp/glm/runs/hy106_comparison_20step/optimization_20261120_round2/REPORT.md) 与 [实现、流水和复现配置](/home/ma-user/work/y00512198/gh_hp/glm/runs/hy106_comparison_20step/optimization_20261120_round2/IMPLEMENTATION.md)。后面的首轮验收表保留为历史对照。

## 开关与集成

|开关|默认|作用|
|---|---|---|
|`HP_FSDP_PACK_REDUCE_INPUT`|`0`|非零 shard dim 的 RS 直接转换/打包到最终 rank-major 布局|
|`HP_FSDP_RS_POOL_BYTES`|`0`|单个归约流水的 RS 输入池字节上限；需同时启用直接打包|
|`HP_MOONEP_FUSED_WGRAD`|未设置|自动读取模型在真实参数上安装的梯度累加标记；`0` 禁用，`1` 显式启用|
|`HP_MOONEP_COMPACT_WEIGHTS`|`0`|按实际副本需求分配权重槽；非融合 home 布局直接复用最终表|
|`HP_MOONEP_REPLICA_PREFETCH`|`1`|FSDP 提前恢复后接续合法保留计划的副本准备；`0` 用于只提前 AG 的消融|
|`HP_MOONEP_ARENA_BUDGET_BYTES`|`0`|megakernel 临时 arena 的扩容预算；0 表示不另设上限|
|`HP_MOONEP_MAX_PENDING_PHASES`|`2`|同一 compute stream 的在途 phase 上限|
|`HP_MOONEP_RETAIN_ARENA_BYTES`|`0`|在反向/步末按预算保留可复用 scratch；0 表示按原策略退役，降低后续 forward 峰值|
|`HP_MOONEP_DIRECT_GRAD_SEND`|`1`|native 真实梯度累加时，副本 FP32 dW 直接写 peer-ordered G send，跳过独立输出与打包复制|
|`HP_FSDP_EARLY_RS_INPUT`|`1`|deferred reduction 流提前退休 RS 输入，保留输出和后续消费流 wait；当前主要作用于 NF|
|`HP_MOONEP_HOME_WEIGHT_VIEWS`|`0`|NF：0 为 stacked 表；1 为借用 home 的 weight-list 单次 GMM；2 为实验性 home/remote 分两次 GMM|
|`HP_MOONEP_RELEASE_W_WORKSPACE`|`0`|NF W unpack 后退还工作区，下一次按当前需求申请|
|`HP_MOONEP_SEPARATE_TOKEN_STREAM`|`0`|实验性 native T 独立 caller stream；W/G 保留原 HCCL 域；本轮不推荐|

开关必须在创建模型/FSDP/runtime 前设置，所有 EP rank 保持一致。RS 池上限按**每个归约流水**计；普通与 deferred 流水分别持有池。它不限制整个进程显存。arena 预算也不包含 FSDP、autograd 保存激活和外部 SHMEM。

`MoonEPExecutor` 与 `SparseMoonEPExecutor` 新增 `gradient_accumulation_fusion: bool | None` 参数。`None` 自动使用 VeOmni 开启 `train_args.gradient_accumulation_fusion` 后在 FSDP 参数安装的 `_veomni_fused_wgrad_owner` 标记；模型关闭该功能且没有显式覆盖时，仍返回常规参数梯度。显式 `True` 可用于不依赖 VeOmni 的 HP 模型适配，但要求 FP32 梯度归约。环境变量是实验覆盖入口，优先于自动检测和构造参数。

首轮 FP32 累加需要包含 `routed_experts_backward_accumulate` 的 native kernel 和 Torch binding；更早二进制无法执行该路径。第二轮 direct-send 沿用同一 ABI 和已构建的二进制，无新增 native 编译。原 `routed_experts_backward_tail` ABI 保留用于消融与既有调用。本机采用独立任务构建目录，未覆盖原实验二进制。

## RS 直接打包与有界输入池

[hsdp_param.py](../../../hyper_parallel/core/fully_shard/hsdp_param.py) 的 `pack_reduce_scatter_input` 按目标 rank 拆分源梯度，直接 `destination.copy_(source)` 到归约 dtype 的最终连续布局，省掉“完整 FP32 cast → 完整 FP32 cat”中的完整 cast 临时区。缩放仍在最终通信输入上执行一次，collective、AVG/SUM 与梯度回写规则保持一致。

[reduce_buffer.py](../../../hyper_parallel/core/fully_shard/reduce_buffer.py) 提供有界、跨参数共享的输入池。只有 `Work.wait()` 完成依赖衔接后才归还槽；跨 stream 重用等待 release event。已租用的输入永不覆盖、不 resize；预算不足使用独立临时分配。RS 输出由后续归约/梯度/优化器持有，本次不把它纳入输入池，以免提前复用仍被读取的梯度。

输入池减少分配次数，却可能增加 forward 常驻显存。应分别比较直接打包、打包加池化，再决定是否组合，不能把减少分配等同于峰值降低。默认预算为零。

## 非融合：home matmul-add 与 BF16 副本回传

[numerics.py](../../../hyper_parallel/distributed/expert_parallel/moonep/numerics.py) 的 `grouped_weight_grad_add` 调用 `npu_grouped_matmul_add_`。在真实参数物理布局上执行 home 的 FP32 累加，不再生成完整 home BF16 dW，也不重复计算 home 的 GMM。

```text
home BF16 X/dY → grouped matmul-add → FSDP FP32 g
replica BF16 X/dY → GMM → 独立 BF16 dW → BF16 EP 回传
                                            ↓
                                owner FP32 g_view.add_(BF16 dW)
```

分组来自冻结的 forward 计划。home 固定前缀与 replica 后缀分别计算；转置的 GLM 参数通过交换操作数直接写物理参数布局。空专家和全空 home token 保留已有 g，不覆盖此前 microstep。

[fsdp_hooks.py](../../../hyper_parallel/distributed/expert_parallel/moonep/fsdp_hooks.py) 通过共享存储和 stride 匹配真实 FSDP 参数，取得 `unsharded_accumulated_grad`。多个远程贡献按 wire/peer 顺序逐份混合类型 add 到 FP32 g，避免先在 BF16 中求和，也不显式生成整份 `.float()` 副本。

自定义 backward 返回 `None` 参数梯度后，显式通知 FSDP 参数就绪门控；不用整份零梯度触发 autograd，不进行第二次累加。归约前完成远程归并；no-sync 下 g 跨 microstep 保留，最终同步仍由 FSDP 控制。

## megakernel：直接累加 home FP32 g

[compute.py](../../../hyper_parallel/core/multicore/modules/routed_experts/compute.py) 为每个实际专家建立 FP32 输出指针表。home 指针指向真实 g；默认 `HP_MOONEP_DIRECT_GRAD_SEND=1` 时，replica 指向通信采用的、按 peer/expert/role 排序的 FP32 G send slice，省去独立 replica dW 及 send-pack copy。关闭此开关才使用独立输出区。转置参数按物理矩阵方向构造任务和 tiling，避免生成整份转置梯度。

[backward.cpp](../../../hyper_parallel/core/multicore/ops/routed_experts/backward.cpp) 的 mapped 路径对 home 使用 CANN `Matmul::IterateAll(g, enAtomic=1)`，对 replica 使用覆盖写 `enAtomic=0`。此参数语义已由本机 CANN 头文件核对，并以非零初始 g、多 microstep NPU 精度探针验证。每个任务仍独占完整输出 tile，token K 维在 cube 内完整归并；原有 g 只在输出时以 FP32 相加。

流水如下：

1. 划分互不重叠的 BF16 dY、FP32 G send/recv，绑定真实 g 与 replica send slice；满足权重、激活与输入导数依赖。
2. 优先生成 replica gate/up、down FP32 dW，发布共同 producer fence。
3. G 流等待 fence 后发送 FP32 replica dW；C 继续 dX，T 按原 tile 协议返回 dX。
4. C 对 home tile 直接 FP32 累加到 g；G 的早期 fence **不代表 home 已完成**。
5. C 在完整 backward tail 后等待远程接收 ready，再按顺序将 FP32 远程贡献加到 g。
6. 通知 FSDP 就绪，等待 bucket 中其他参数后提交原生 RS/AR；no-sync 只保留累计值。

已有 home g 不放进通信载荷，也不生成 BF16 owner 梯度输出。写 g 的远程 add 排在 home native 写之后；通信 staging 与 g 分离。只允许精确对应的 send slice 别名；dY、recv、其他 fragment 和 home g 均不可重叠。mapped gradient 的底层校验按字节检查相邻 BF16/FP32 dense 区域；其他计算接口仍拒绝共享输入 storage。

## 动态序列内存

非融合 `HP_MOONEP_HOME_WEIGHT_VIEWS=0` 保持原来的 stacked 表：按实际 `home + replica` 数量、每四个专家一个容量桶增长，home 连续化直接写表前缀。模式 `1` 借用 FSDP home 物理参数的转置视图，仅分配 remote 表，并用 CANN weight-list 单次 GMM；当前实现校验 home 与副本准入总数不超过 128。模式 `2` 也借用 home，但按 home/remote 两个 batch 执行 GMM；只有一部分非空时保持一次 GMM，两部分均非空时需要额外 launch 和输出拼接，须独立实测。两个模式均覆盖转置、零 token 专家与反向 dX。

设置 `HP_MOONEP_RELEASE_W_WORKSPACE=1` 后，W 解包提交完成即可退还工作区引用，下一次仅按当前需求申请，不无条件恢复历史高水位。通过原 W 流顺序、ready/last-reader 事件和 allocator stream 记录保护在途存储，禁止 resize 尚未完成的通信输入。保留模式默认为 `0`，便于单独消融。

设置 `HP_FSDP_EARLY_RS_INPUT=1`（默认）后，FSDP 在 deferred reduction 流提交 post-backward 后建立 Work wait 依赖，提前退休 RS 输入；保留输出和 Work，让后续输出消费流再次 wait。跨流输入通过 `record_stream` 保护，池租约与非池存储分别回收；不增加、删除或改变 collective 顺序。本实验非融合有独立 deferred 队列；native 在自定义 backward 内完成归并，没有这条队列，因此不能把 native 开关对照中的波动归因于此优化。allocator 的逻辑释放与设备端最终可复用必须区分。

megakernel BF16 replica pool 也按实际需求、每四个 replica 一桶增长。分配失败在提交权重载荷前通过 EP control group 收敛，避免某一 rank 失败而其他 rank 进入数据通信。retire/restore 保留实际容量，未使用的槽不预留整个最大专家表。

现有 [PhaseArena](../../../hyper_parallel/core/multicore/modules/routed_experts/arena.py) 已复用激活、token 和临时区域，本次补充扩容预算、在途 phase 上限及 `memory_snapshot()` 账本：容量/有效字节、storage 身份、增长次数、旧/新并存的扩容峰值与尚未完成的旧区退役计数。不能把 autograd 保存激活或 FP32 g 放入短期 arena。

`capacity_growth_count` 区分真正的容量增长，`restoration_count` 记录主动退役后按原容量恢复，避免把每次重新申请都误报成动态序列扩容。关闭 direct send 时独立 replica dW 区也在反向结束退役；启用时不再申请该区域。可用 `HP_MOONEP_RETAIN_ARENA_BYTES` 对少量 scratch 做有界保留，必须另测其分配收益和 forward 常驻代价；默认不保留，不将这个可选策略自动合入低内存组合。

容量按真实 token selection 数量估算：`S=T×topK`，接收端按实际 R 分配。12K packing、topK8 的 S 可接近 98K。固定一条 8K sequence 的池不能覆盖本实验，也不能靠截断序列节省显存。已有 2 MiB 字节容量桶在不同 shape 间保留高水位；预算不足在 collective admission 前失败，不丢 token、不跳梯度、不每步 empty_cache。

### 模型输出也要进入内存账本

均衡专家 token 不会削减其他模块的输出。在 HY 的全 ignored AR 标签分支中，默认返回 logits 会生成全词表矩阵，随后仅通过 `0 * logits.sum()` 连接零梯度。chunk CE 只覆盖有监督分支，不能消除这块输出及其反向零梯度载荷；若 trainer 持有整个 output，logits 也会继续存活。

训练端不读取 logits 时，VeOmni 适配应显式传入 `return_logits=False`，在无 AR 监督分支用单行 `lm_head` keepalive，并继续将其零倍和连接到 loss。空文本可用同 dtype/device 的一行零输入。这保留 head 的 FSDP 调用和非 None 零梯度，使已有优化器动量和 weight decay 继续更新；直接跳过 head 不具有相同语义。显式请求 logits 的接口、监督 CE 和 teacher 路径应单独保留。

内存验证要分别检查输出存活、arena、W/G、FSDP RS 与 allocator current 状态。不同时间的峰值不能相减推断瞬时碎片，`arena.reserved_bytes=0` 也不表示当前 autograd frame 已销毁全部借用视图。局部 event wait 只有在独立消融证明有益后才可采用，不能用热点同步替代生命周期归因。

## 显式 FSDP prefetch=2

HP 的 [prefetch.py](../../../hyper_parallel/core/fully_shard/prefetch.py) 提供模型子图的执行预取计划接口。调用者明确提供 block 顺序和子单元的真实 forward 顺序，未选入序列的 embedding/视觉等单元保留 demand loading 并列入审计结果。

本机 VeOmni 旧构建器要求单元组集合与 block 序列完全相等，并使用专家先于 router 的注册顺序；直接设环境变量会因额外 embedding/冻结视觉组而启动失败。本轮任务 runner 将构建入口接到上述 HP 接口，并按 HY 实际调用顺序将 MoE 单元组声明为 `[block, router, experts]`。实际适配器保存在 HP 仓的 [run_hy106.py](../../../examples/expert_parallel/run_hy106.py)，可作为 torchrun worker 替代直接 `-m tasks.transfusion.train_hy`；实验 runner 调用同一个适配器并添加审计。无需修改 VeOmni 仓库；审计文件保存完整计划及 demand-loaded 单元。其他模型也必须由适配器给出真实顺序，不能从模块注册顺序猜测。

启动时使用 HP 仓内入口，配置和数据继续使用既有 VeOmni 环境。后端通过 YAML 参数传入：

```yaml
train_args:
  data_parallel_mode: hp_fsdp2
  moonep_variant: megakernel  # nonfused 为非融合；null 保持普通 EP
```

`moonep_variant` 也可作为 `build_parallelize_model(..., moonep_variant="megakernel")` 的显式参数。
不读取后端选择环境变量。megakernel 使用 HP 标准安装/构建目录中的 native 库；先完成该组件的构建和环境初始化，
再确认 Python editable 指向本 HP checkout 后执行：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=16 \
  /path/to/hyper-parallel/examples/expert_parallel/run_hy106.py --config /path/to/config.yaml
```

在该入口接线后设置：

```bash
export VEOMNI_FSDP_PREFETCH_STRATEGY=execution
export VEOMNI_FSDP_FWD_PREFETCH=2
export VEOMNI_FSDP_BWD_PREFETCH=2
```

depth 按展开后的 FSDP unit 计，不等于两个 decoder layer。仅配置 `hy_prefetch_depth=2` 不保证入口生效。普通 EP 对照也应使用同样的有效计划。

HP scheduler 先提交全部目标 AG，再调用 `complete_prefetch()`，由共享 post-unshard stream 衔接 `fsdp_post_unshard → runtime.prepare → replica ready`。消费处只等待自己的 ready。原始前向的未来 router plan 通常尚未产生，因此只能先提前 home AG；重算/反向才可利用当前迭代保留计划提前副本交换，不能使用上一步路由。

验收要导出 FQN targets、`fsdp_ep_prefetches`、最大在途 lease，以及 trace 中真实 AG/副本提前区间和 consume 等待。设置 `HP_MOONEP_REPLICA_PREFETCH=0` 对比“相同 depth=2、仅提前 AG”，隔离副本预取的收益。若 depth=2 超预算或未覆盖专家，应据实报告，不能用 depth=1 结果冒充完成。

## 验证

CPU UT 覆盖直接 RS 布局与类型转换、池租约/预算、空专家、多 microstep、远程 BF16 逐份加入 FP32 g、None 梯度就绪通知、按需权重容量及 home 布局别名。

NPU 数值 worker 为 [精度探针](../../../tests/torch/expert_parallel/_test_moonep_accumulation_precision.py)，显式传入新 native 目录，与 CPU FP64 dW oracle 比较，覆盖转置布局、非零 g、多次累加与 replica 覆盖写。BF16 dX 另行对比 FP64 矩阵乘参考。训练 loss/norm 漂移作为观测指标，不能代替该精度验收。

性能每组 20 step，第 4 step 采一个 step，不带 host stack，保留显式 Python ranges；统计第 11–20 step 全 rank 最大墙钟的均值和标准差。同时报告 allocated/reserved 跨 rank 峰值、retry 增量、有效 token 数和 loss/norm 全精度记录。失败或少于 20 step 的结果不进入稳态均值。本轮正式组合已完成相同口径的三次无 profiler 重复和长短序列切换验证，结果见下节。

<!-- HY106_VALIDATION_RESULTS_START -->
## 首轮实施与验收记录（2026-11-20）

下表为 16 rank、EP2、专家 FSDP8、3 decoder layer 约 7.20B 减层模型；动态 12K packing，三次独立无 profiler 运行，每次 20 步，取 11–20 步的全 rank 最大墙钟均值。普通 EP 对照使用相同的有效 FSDP forward/backward prefetch=2。

|实现|三次均值 ± 运行间标准差，s|相对普通 EP|11–20 allocated 峰值，GiB|1–20 allocated 峰值，GiB|1–20 retry 合计|
|---|---:|---:|---:|---:|---:|
|VeOmni EP + HP FSDP2|2.412666 ± 0.021497|1.0000×|29.306|29.306|0|
|MoonEP 非融合：P/G/W/PF2|2.582600 ± 0.023743|1.0704×|35.626|35.763|0|
|MoonEP megakernel：P/G/W/PF2|2.408926 ± 0.010557|0.9984×|32.310|34.251|0|

P=RS 直接打包；G=真实 FP32 g 直接累加；W=实际副本容量与 home 布局复用。megakernel 与普通 EP 的耗时差在重复波动范围内；非融合仍有额外耗时。主配方保持 RS 常驻输入池、arena 额外保留关闭；这些可选开关已单独消融，不因个别加速而默认增加显存。

FP32 精度探针、实际 H4096/I1408 大形状探针，以及四 rank EP2/FSDP2 的多 microstep/no-sync、重算、空专家、topK2、router score 与无迁移测试均通过。相关 CPU UT 为 388 passed。非融合 home dW 相对 FP64 的误差约 7e-8，保留 BF16 replica wire 后的分布式 dW 约 1e-3；megakernel 分布式 FP32 dW 约 4e-7–6e-7。普通 EP 自身已有 FP32 wgrad，不能把减少旧 MoonEP 中间舍入误称为精度超过普通 EP。

训练 loss/norm 全部有限且无跳步；非融合第二次重复的 norm 漂移高于普通 EP 自身波动，完整记录保留。固定输入模块正确性不等同于长期收敛证明。Torch allocated/reserved 也不完整包含驱动/HCCL/SHMEM；1–20 指训练步，模型构建阶段的峰值未单独采样。

推荐先在训练配置启用 `gradient_accumulation_fusion=true`，并使用：

```bash
unset HP_MOONEP_FUSED_WGRAD  # 自动读取 VeOmni 的真实参数标记
export HP_FSDP_PACK_REDUCE_INPUT=1
export HP_MOONEP_COMPACT_WEIGHTS=1
export HP_FSDP_RS_POOL_BYTES=0
export HP_MOONEP_RETAIN_ARENA_BYTES=0
export HP_MOONEP_REPLICA_PREFETCH=1
export VEOMNI_FSDP_PREFETCH_STRATEGY=execution
export VEOMNI_FSDP_FWD_PREFETCH=2
export VEOMNI_FSDP_BWD_PREFETCH=2
```

VeOmni 自动接线已用不设置环境覆盖的 16 rank 实际训练验证；两种后端所有 rank 的有效值均为 true。副本 retained-plan 预取的计数、真实 FQN 目标与无重复物化已核对；设备重叠和整步收益分开报告，不能只因计数生效就声称权重通信完全隐藏。本轮重采样和轻采样均未观察到 W payload 与所选重计算核重叠；当前 W/G/T 共用 EP 提交流，独立 W 队列/数据域仍需后续验证。

完整 [实验报告](/home/ma-user/work/y00512198/gh_hp/glm/runs/hy106_comparison_20step/optimization_20261119/REPORT.md)、[逐项收益](/home/ma-user/work/y00512198/gh_hp/glm/runs/hy106_comparison_20step/optimization_20261119/factor_effects.csv)、[精度与性能数据](/home/ma-user/work/y00512198/gh_hp/glm/runs/hy106_comparison_20step/optimization_20261119/final_results.json)、[代码补丁](/home/ma-user/work/y00512198/gh_hp/glm/runs/hy106_comparison_20step/optimization_20261119/source.patch) 保存在本工作区。完整 106B、更大 EP、设备化规划、迁移经济性门槛及额外 loss 层融合仍是后续项。

<!-- HY106_VALIDATION_RESULTS_END -->
