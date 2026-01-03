# cem-rag（设计规格 v2）

> **这是 cem-rag 的“完整设计规格文档（Design Spec）”。**  
> 本文档目标不是“快速上手”，而是**冻结架构、边界与不变量**，用于：
> - 驱动 Cursor / Codex 等代码代理从 0 实现仓库
> - 作为后续论文、系统演进的一致性参考
>
> ⚠️ 若与 README.md（实现版）存在冲突，以本文件为准。

---

## 0. 项目目标与问题定义

cem-rag 的目标是验证并工程化以下思想：

> **文档可以在“压缩域”直接完成向量化与检索，  
> 仅在被召回后才进行局部解压，从而显著降低存储与 I/O 成本。**

与传统 RAG 的根本区别：
- 传统：text → embedding → index → retrieve → generate
- cem-rag：compressed bytes → embedding → index → retrieve → *late decompression* → generate

---

## 1. 全局不变量（Hard Constraints）

以下约束在任何实现、任何优化中都 **必须成立**：

### 1.1 压缩域不变量

- **MUST NOT** 在索引、检索、评估阶段解压文档
- **MUST** 使用压缩字节流作为 CEM 的输入
- 解压 **ONLY ALLOWED** 在 Top-K 召回之后

### 1.2 解耦原则

- zpkg **不感知 corpus-id**
- zpkg **不感知 BEIR / RAG / 训练 / 评估**
- 语义映射属于上层业务模块

### 1.3 Query 原则（已冻结）

- Query 天生是用户输入文本
- **Query Encoder MUST be text-based**
- 使用 CEM 编码 query 被明确视为 **Non-Goal**
  - 原因：对 query 再压缩一次在语义与工程上均无意义

---

## 2. zpkg：压缩容器模块（Compression Container）

### 2.1 定位

zpkg 是一个：
- 二进制
- 稳定
- 与任务无关

的压缩数据随机访问容器。

它的存在是为了回答一个问题：
> **“如何在不解压整个文件的情况下，O(1) 访问任意文档的压缩表示？”**

---

### 2.2 二进制布局（冻结）

```
[ Header (32 bytes) ]
→ [ Dictionary Section (zstd shared dict) ]
→ [ Index Table (uint32 offsets) ]
→ [ Chunk 0 ]
→ [ Chunk 1 ]
→ ...
→ [ Chunk N-1 ]
```

说明：
- Index Table 是 uint32 offset 数组
- chunk_i 的长度 = offset[i+1] - offset[i]
- 最后一个 chunk 使用 file_size 推导

---

### 2.3 zpkg 明确的 Non-Goals

- ❌ corpus-id / 文档 ID
- ❌ 文档元数据
- ❌ 文本语义
- ❌ 评估 / RAG 逻辑

---

### 2.4 语义映射（外挂）

语义映射 **不属于 zpkg**，而是：

```
corpus-id → (zpkg_path, chunk_index)
```

推荐存储形式：
- pickle / json / sqlite（v1 推荐 pickle）

---

### 2.5 冻结 API

#### 构建

```python
build_zpkg(corpus_jsonl_path, output_dir) -> ZPKGManifest
```

#### 压缩态读取（核心接口）

```python
get_compressed_chunk(zpkg_path, corpus_id | chunk_index) -> bytes
```

#### 解压读取（受限接口）

```python
get_decompressed_chunk(zpkg_path, corpus_id | chunk_index) -> str
```

---

## 3. CEM：Compression Embedding Model

### 3.1 总体目标

学习一个映射：

```
compressed_bytes  →  semantic_embedding
```

使得该 embedding 在检索空间中与文本 embedding 对齐。

---

### 3.2 Student：BLT（Byte Latent Transformer）

- 基于：https://github.com/facebookresearch/blt
- 使用路径：
  - byte-level encoder
  - local encoder
  - global transformer
- **明确移除 / bypass decode 与预测路径**

#### 实现建议（非强制，但推荐）：
- 使用 wrapper，而非大幅 fork BLT
- 在 forward 中暴露 encoder-only 路径

---

### 3.3 Teacher：Qwen-0.6B / E5

- Teacher 输入：解压后的原始文本
- Teacher embedding 定义：
  - last_hidden_state
  - attention-mask mean pooling
- 输出维度：**1024**

#### Teacher Cache（重要）
- Teacher embedding SHOULD be cached
- 推荐磁盘级 cache（npy / memmap）
- 防止 teacher 成为训练 / 评估瓶颈

---

### 3.4 Embedding 对齐规范（冻结）

- CEM embedding 维度：**1024**
- 与 teacher embedding 处于同一向量空间
- **MUST L2 normalize** 后再进入 ANN

---

### 3.5 Loss 设计（语义级定义）

对同一 corpus-id：

- **Distillation Loss**
  - cosine / MSE on normalized embeddings
- **Contrastive Loss（InfoNCE）**
  - 正样本：(student_i, teacher_i)
  - 负样本：batch 内其它样本
- **Triplet Loss**
  - anchor：student_i
  - positive：teacher_i
  - negative：teacher_j (j ≠ i)

```
L = α·L_distill + β·L_contrastive + γ·L_triplet
```

---

## 4. Evaluation：BEIR（模态无关）

### 4.1 设计哲学

> **Evaluator MUST be modality-agnostic**

评估器：
- 不区分文本 / 压缩
- 只认 corpus-id

---

### 4.2 Encoder 抽象

```python
encode_corpus_ids(corpus_ids) -> embeddings
encode_queries(texts) -> embeddings
```

- Doc Encoder：
  - CEM：压缩 bytes
  - Text 模型：解压文本
- Query Encoder：
  - 始终使用 text-based encoder（Teacher / E5）

---

### 4.3 Benchmark

- 标准 BEIR 流程
- 指标：nDCG@10 / MRR@10 / Recall@K

---

## 5. RAG（LangChain）

### 5.1 Indexing（首端，压缩入库）

- **MUST NOT** 使用 add_texts()
- **MUST** 使用 CEM embedding 构建索引

流程：

```
corpus-id
 → compressed bytes (zpkg)
 → CEM embedding
 → GPU VectorStore (Faiss-GPU)
```

---

### 5.2 Query-time RAG（尾端）

```
query (text)
 → query encoder
 → ANN search
 → Top-K corpus-id
 → late decompression
 → LangChain prompt
 → LLM
```

---

## 6. 仓库结构（冻结）

```
cem-rag/
├── src/
│   ├── zpkg/
│   ├── cem/
│   │   ├── models/
│   │   ├── training/
│   │   └── encoding/
│   ├── eval/
│   └── rag/
│       ├── indexing/
│       ├── retrieval/
│       └── chain.py
├── tests/
├── data/
├── output/
├── scripts/
├── configs/
├── README.md
├── requirements.txt
└── setup.py
```

---

## 7. 明确 Non-Goals

- ❌ query 压缩
- ❌ zpkg 存语义信息
- ❌ 索引阶段解压
- ❌ end-to-end 生成式训练

---

## 8. 给代码代理（Cursor / Codex）的执行说明

- 本文件是**最高优先级设计规范**
- 不得引入未定义模块
- 不得弱化任何 MUST / MUST NOT
- 允许实现细节差异，但不得破坏不变量

---

## 9. 开发指南

### 9.1 项目结构

本项目采用 **src 布局**：

```
cem-rag/
├── src/              # 所有源代码包
│   ├── zpkg/        # ZPKG 压缩格式
│   ├── eval/        # 评测框架
│   ├── cem/         # Compression Embedding Model（未来）
│   └── rag/         # RAG 组件（未来）
├── tests/            # 测试脚本
├── data/             # 数据文件
├── output/           # 输出文件（gitignored）
├── scripts/          # 工具脚本
├── configs/          # 配置文件（未来）
├── README.md         # 主文档
├── requirements.txt  # Python 依赖
├── setup.py          # 包配置
└── .gitignore        # Git 忽略规则
```

### 9.2 开发环境设置

#### 标准安装方式（推荐）

```bash
# 1. 创建虚拟环境（推荐使用 Python 3.11）
python3.11 -m venv venv
source venv/bin/activate  # 或 Windows: venv\Scripts\activate

# 2. 安装项目（开发模式）
pip install -e .

# 3. 安装评估依赖（如果需要运行 LongEmbed 评估）
pip install -e ".[eval]"
```

这会让所有包可以从任何地方导入，并且以开发模式安装，代码修改会立即生效。

#### 可选：安装开发工具

```bash
pip install -e ".[dev]"
```

#### 运行测试

```bash
# 测试脚本会自动添加 src 到路径
python3 tests/test_random_decompress.py
python3 tests/test_mapping_exact.py
```

#### 运行评测

```bash
# 评测脚本会自动添加 src 到路径
python3 src/eval/longembed_eval.py --model Qwen/Qwen3-Embedding-0.6B
python3 scripts/download_model.py
```

### 9.3 导入规范

- **src/ 内部**：使用相对导入或绝对导入（如 `from zpkg.reader import ZPKGReader`）
- **从 tests/ 或 scripts/**：脚本会自动添加 `src/` 到 `sys.path`，使用 `from zpkg.reader import ZPKGReader`
- **安装后（`pip install -e .`）**：所有导入在任何地方都正常工作

### 9.4 添加新模块

1. 在 `src/` 中创建模块（如 `src/cem/`）
2. 添加 `__init__.py` 使其成为包
3. 如需要，更新 `setup.py`
4. 在 `tests/` 中添加测试

---

## 10. LongEmbed 评测框架

### 10.1 LongEmbed Benchmark

LongEmbed benchmark 来自论文 ["LongEmbed: Extending Embedding Models for Long Context Retrieval"](https://arxiv.org/pdf/2404.12096)，用于评测长上下文检索场景下的 embedding 模型性能。

包含的任务：
- **2 个合成任务**：Passkey, Needle
- **4 个真实任务**：NarrativeQA, QMSum, SummScreenFD, 2WikiMultihopQA

这些任务旨在测试 embedding 模型处理长上下文检索的能力。

### 10.2 安装依赖

#### 标准安装方式（推荐）

使用标准的 Python 包安装方式：

```bash
# 1. 创建虚拟环境（推荐使用 Python 3.11）
python3.11 -m venv venv
source venv/bin/activate  # 或 Windows: venv\Scripts\activate

# 2. 升级 pip
pip install --upgrade pip setuptools wheel

# 3. 安装项目（开发模式，包含评估依赖）
pip install -e ".[eval]"
```

这会自动安装所有核心依赖和评估依赖（包括 mteb）。

#### 如果 LongEmbed 任务不可用

如果安装的 mteb 版本不支持 LongEmbed，可以从源码安装：

```bash
pip install git+https://github.com/embeddings-benchmark/mteb.git
```

#### 旧版安装脚本（已废弃）

> ⚠️ **注意**：`scripts/setup_venv.sh` 和 `scripts/setup_eval.sh` 已废弃，推荐使用上述标准安装方式。

### 10.3 验证安装

```bash
# 激活虚拟环境（如果使用）
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ./venv

# 设置 HuggingFace token（推荐，避免限流）
export HF_TOKEN=your_token_here
# 获取 token: https://huggingface.co/settings/tokens

# 下载并验证模型
python scripts/download_model.py Qwen/Qwen3-Embedding-0.6B

# 检查 mteb LongEmbed 支持
python -c "from mteb.tasks import LongEmbedRetrieval; print('LongEmbed available')"
```

### 10.4 使用方法

#### 基本评测

在所有 LongEmbed 任务上评测模型：

```bash
python3 src/eval/longembed_eval.py --model Qwen/Qwen3-Embedding-0.6B
```

#### 自定义评测

```bash
# 评测特定任务
python3 src/eval/longembed_eval.py \
    --model Qwen/Qwen3-Embedding-0.6B \
    --tasks LongEmbedRetrieval/Passkey LongEmbedRetrieval/Needle \
    --output-dir results/qwen_0.6b \
    --device cuda \
    --batch-size 32
```

#### 参数说明

- `--model`: HuggingFace 模型名称或路径（默认：`Qwen/Qwen3-Embedding-0.6B`）
- `--output-dir`: 结果保存目录（默认：`results/{model_name}`）
- `--tasks`: 要评测的特定任务（默认：所有 LongEmbed 任务）
- `--device`: 运行设备（`cuda` 或 `cpu`，默认：`cuda`）
- `--batch-size`: 编码批次大小（默认：32）

### 10.5 预期结果

根据 LongEmbed 论文，参考模型的基线结果：

- **E5-Mistral**: ~64.4 平均分数
- **E5-Mistral + NTK (32k)**: ~75.3 平均分数
- **BM25**: ~90.4 平均分数

我们将对比 Qwen3-Embedding-0.6B 的结果与这些基线，以验证评测框架是否正确。

### 10.6 输出

结果保存在输出目录中：
- `results_summary.json`: 详细的评测结果
- 任务特定的子目录，包含详细指标

### 10.7 注意事项

- 评测框架设计为模态无关（符合项目设计规范）
- 结果应可复现，并与论文基线可比
- 此框架将用于评测基于文本和基于压缩的 embedding
