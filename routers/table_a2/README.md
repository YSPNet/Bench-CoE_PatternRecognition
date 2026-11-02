# Table A2 补跑

独立补跑全部 13 种原论文方法：Oracle、Strongest、Cheapest、KNN、PRkNN、OVR、K-means、Linear、MLP、CosineCls、RouterDC、ZOOTER、VLC。所有专家答案、正确性和成本来自现有缓存，不调用专家模型或远程模型 API。

协议复用 `outputs/bench_coe_query/ood/protocol/HELD_OUT_PROTOCOL.json` 及四份 JSONL，严格校验哈希、样本 ID、矩阵行号和原始 split 的交集。留出 MMMU_DEV_VAL、MathVision_MINI、ChartQA_TEST。当前目录的四个视图是 18,665 / 2,680 / 5,341 / 767 条。原始 SPLITS、Y/C 和历史实验目录保持只读。

本实验属于“按论文规则重建的 held-out 实验”。没有作者原始 OOD 样本清单，不能声称逐数复现 Table A2。BenchCoE 的训练标签和 checkpoint 规则不用于这些原论文方法。

## 命令

在仓库根目录执行。默认配置为 `configs/table_a2.yaml`，输出在 `outputs/table_a2`。命令失败时返回非零退出码。

```bash
# 检查全部方法的输入、特征文件和离线骨干缓存
python scripts/table_a2/run.py preflight

# 先运行不需要特征/骨干的三个基线
python scripts/table_a2/run.py run --methods oracle strongest cheapest

# 缺少原特征时，抽取 BGE-M3 + DINOv2；按批次断点续跑
python scripts/table_a2/run.py extract-features

# 补齐 Table A2 缺少的九种方法
python scripts/table_a2/run.py run --methods strongest cheapest knn prknn ovr kmeans linear cosinecls zooter

# 用同一协议补跑全部方法，自动复用已完成候选和冻结预测
python scripts/table_a2/run.py run

# 重新从冻结预测计算指标，不训练模型
python scripts/table_a2/run.py evaluate

# 汇总全部已配置方法；缺项时报错
python scripts/table_a2/run.py aggregate

# 查看阶段性结果，并列出缺项
python scripts/table_a2/run.py aggregate --allow-partial
```

`--methods` 和 `--seeds` 选择本次处理的子集，不改变配置中声明的完整实验。确定性方法只运行配置中的第一个 seed；Linear/MLP/CosineCls/RouterDC/ZOOTER/VLC 默认运行五个 seed。每个随机方法、每个 seed 独立扫描六个 λ，按 `dev_seen` Rank Score 选择；分数相同保留配置中的第一个 λ。ID 与 OOD 使用同一个已保存的 checkpoint。

CPU 运行可以使用 `--device cpu`。设备、超参数、编码器、配置和代码哈希会冻结；之后需要改变这些设置时，应指定新的 `--output-dir`，不要复用已经冻结的目录。例如：

```bash
python scripts/table_a2/run.py run --methods oracle strongest cheapest --device cpu --output-dir /tmp/table_a2_cpu
```

要修改 λ、epoch、特征或方法参数，复制 `configs/table_a2.yaml` 到新配置，修改后用 `--config` 指定，并选择新输出目录。通过 `CUDA_VISIBLE_DEVICES` 选择实际使用的 GPU。

## Factory 环境与并行执行

本机执行环境位于 `.venvs/table_a2_factory`，继承 Factory 的 PyTorch，独立安装 transformers 4.51.3、sentence-transformers 3.4.1 和 huggingface-hub 0.36.2。没有修改 Factory 的包。

```bash
# 优先复用本地 BGE 和 BERT；仅下载其余三个公开骨干
.venvs/table_a2_factory/bin/python scripts/table_a2/prepare_weights.py \
  --cache-dir .cache/table_a2/huggingface/hub \
  --manifest outputs/table_a2_setup_20260914/PRETRAINED_MODELS.json \
  --local-bge-m3 /home/sm5/ys/PR/Y/models/bge-m3 \
  --local-model bert-base-uncased=/home/sm5/ys/PR/Y/models/bert-base-uncased

# 抽取/复用特征、准备图片、检查输入，再启动全部方法/种子任务
.venvs/table_a2_factory/bin/python scripts/table_a2/execute.py \
  --config configs/table_a2_factory_v2_20260914.yaml \
  --gpus 3 1 2 4 5 6 7 --cpu-workers 2
```

执行器按 method/seed 分配 GPU，各任务内部完整扫描六个 λ。任务写入独立的模型、预测与指标目录，不并发覆盖总表。`EXECUTION.json` 记录准备阶段、任务 PID、GPU、日志和退出状态；总表定期汇总并保留缺项，只有全部任务通过、完整汇总成功才标记 complete。重启执行器会核验并复用已经提交的候选与预测。`prepare-images` 也可独立执行；`run --worker --methods <method> --seeds <seed>` 要求端到端图片缓存已经完整准备好。

预训练清单固定下载版本或本地内容哈希。本地模型只通过缓存符号链接读取，不复制或改写源权重。VLC 原实现另存 `model_model.pt`，补跑同时绑定这份权重和 `model.pkl`，并在读取测试结果前验证两者。

## 特征与图像

特征默认读取 `EMBEDDINGS/text/bge-m3.parquet` 和 `EMBEDDINGS/vision/dinov2-base.parquet`。允许复用既有特征，但要求必要样本覆盖完整、ID 唯一、向量有限。按 sample_id 对齐，禁止均值或零向量补缺。抽取时不使用 Y/C，使用已有本地编码器接口，API 编码器被拒绝。完成的批次签名和文件哈希保存在 `feature_shards/`，不同输入不会混合。

默认 `runtime.offline: true`，强制 Hugging Face/Transformers 离线加载。缺少编码器或骨干权重时，先将对应本地缓存准备好；需要下载开源骨干时，可在新配置中显式设置 `offline: false`。这只涉及开源路由骨干，程序没有专家答案再生成或模型 API 调用入口。

按数据集 TSV 的原始 key 解码图片，跟随引用，拒绝缺图和坏图。MMMU 的多图样本按官方 TSV 顺序解码，避免缓存 JSONL 中被拆开的路径列表：特征方法复用原特征接口的均值后归一化，端到端方法与原 Dataset 一样使用第一张图，并记录图片数量、选中序号。首次运行把必要视图的首张图片缓存为 RGB PNG 到新输出目录的 `images/`。后续核验缓存文件哈希，原 Router 接收可读图片路径，避免原加载器产生黑图替代。PNG 缓存需要额外磁盘空间。BenchCoE 的单图解析器未修改，补跑仅复用其低层解码和 TSV 索引。

## 训练行为与输出

KNN/PRkNN/OVR/K-means 直接复用现有 Router 的 fit/predict。端到端方法直接复用原 Router 和损失函数，只传入 train_seen/dev_seen。

CosineCls 显式使用 `lambda_sample_sample: 0.0`，与论文附录 G.1 的 Sample-LLM 单项损失保持一致；RouterDC 保留 Sample-Sample 权重 1。开源 CosineCls 的默认值也是 1，直接采用该默认值会使两种方法的目标相同。旧配置的实际训练保留在 `outputs/table_a2_factory_20260914` 并标记为 superseded，当前完整补跑使用 `outputs/table_a2_factory_v2_20260914`。

Linear/MLP 复用原网络与标签构造，保留原成本缩放（Linear 为 1，MLP 为 100），用统一 minibatch AdamW、cosine scheduler 和真实软标签交叉熵训练。这避免原 Linear CPU 路径把软标签 argmax 为硬标签。该训练实现及超参数会记录，不声称与原脚本每个优化细节完全相同。λ=∞ 使用确定性的 cheapest-correct one-hot 极限；共享软标签函数已修复 inf×0 的 NaN。

Strongest 从 train_seen 的平均正确率选择固定模型；Cheapest 从 train_seen 的平均成本选择固定模型。禁止使用原 baseline 入口的 test-fit 或原 sweep 的 test-rank 选参。Oracle 明确标记为可见答案的理论上界，其预测由评测阶段生成。

每个候选的验证成绩、模型哈希和验证预测保存在 `runs/<method>/seed_<s>/lambda_<λ>/`。最优 λ 和 checkpoint 绑定记录在 `selection.json`。所有请求范围内的实际 Router 预测先保存至 `predictions/`，再读取目标 Y/C 计算指标；模型、预测或输入改变后拒绝继续。

汇总输出：

- `summary/table_a2.csv`、`table_a2.md`：ID/OOD 均值和标准差。
- `summary/per_seed.csv`：每个 seed 的指标、选定 λ 和预测签名。
- `summary/per_dataset.csv`：各数据集的完整指标。
- `summary/COVERAGE.json`：本次汇总范围、完成状态和缺项。

Accuracy 和 Rank Score 乘 100；Cost 乘 10,000，单位为美元/万样本。保留全部测试样本，包括所有专家均答错的样本。每个 seed 在完整视图上先求平均准确率、平均成本，再计算 Rank Score；不平均数据集 Rank Score。所有方法与 ID/OOD 使用同一份固定成本边界。标准差为样本标准差（ddof=1）；确定性方法只运行一次。

## 验证

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/table_a2
```

测试包含不同矩阵行列顺序、held-out 训练泄漏、inf/零成本、真实特征方法训练与重载、软标签 CPU 训练、冻结预测篡改、断点续跑、严格图像/特征抽取及端到端适配接口。端到端接口测试使用本地替身，不下载大模型；真实骨干运行仍需对应权重。
