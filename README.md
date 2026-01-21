# CEM-RAG: Compression Embedding Model for Retrieval-Augmented Generation

> **压缩域向量化与检索系统**  
> 在压缩域直接完成文档向量化，仅在被召回后进行局部解压，显著降低存储与 I/O 成本。

---

## 目录

### 第一部分：项目目的与研究意义
- [1 项目目的与研究意义](#1-项目目的与研究意义)

### 第二部分：使用手册
- [2 快速开始](#2-快速开始)
- [3 核心功能使用](#3-核心功能使用)
- [4 常见问题（FAQ）](#4-常见问题faq)

### 第三部分：架构与模块详解
- [5 整体架构与设计原则](#5-整体架构与设计原则)
- [6 Zpkg 压缩容器模块](#6-zpkg-压缩容器模块)
- [7 Encoders 编码器模块](#7-encoders-编码器模块)
- [8 Train 训练模块](#8-train-训练模块)
- [9 Eval 评估模块](#9-eval-评估模块)
- [10 RAG 应用模块](#10-rag-应用模块)

### 第四部分：开发指南
- [11 开发环境设置](#11-开发环境设置)
- [12 AI 代理开发指南](#12-ai-代理开发指南)

### 第五部分：附录
- [13 参考资源](#13-参考资源)

---

## 1 项目目的与研究意义

#### 1.1 核心问题

传统 RAG（Retrieval-Augmented Generation）系统面临的核心挑战：

1. **存储成本高**：大规模文档库需要存储大量原始文本和对应的 embedding
2. **I/O 开销大**：索引构建和检索过程中需要频繁读取和解压文档
3. **扩展性受限**：随着文档库规模增长，存储和 I/O 成本线性增长

#### 1.2 核心思想

**CEM-RAG 的核心创新**：在压缩域直接完成向量化与检索，仅在被召回后才进行局部解压。

```
传统 RAG：
text → embedding → index → retrieve → generate

CEM-RAG：
compressed bytes → embedding → index → retrieve → late decompression → generate
```

#### 1.3 研究意义

1. **降低存储成本**：压缩域存储相比原始文本可节省 70-90% 空间
2. **减少 I/O 开销**：索引和检索阶段无需解压，显著提升吞吐量
3. **提升扩展性**：支持更大规模的文档库，降低部署成本
4. **保持检索质量**：通过知识蒸馏，压缩域 embedding 与文本 embedding 在检索空间中对齐

#### 1.4 技术路线

- **zpkg**：基于 zstd 的自研压缩容器格式，支持 O(1) 随机访问任意压缩 chunk
- **BLT**：Byte Latent Transformer，作为 CEM 的核心基础模型
- **Qwen-Embedding**：作为教师模型，用于知识蒸馏训练 CEM
- **CEM**：Compression Embedding Model，基于 BLT 学习压缩字节到语义向量的映射
- **pgvector**：PostgreSQL 向量扩展（支持最低 PostgreSQL 12），用于向量存储和检索
- **pytest**：单元测试框架

---

## 2 快速开始

#### 2.1 环境要求

- Python >= 3.8（推荐 3.11）
- CUDA（推荐，用于 GPU 加速）

#### 2.2 安装

```bash
# 1. 克隆仓库
git clone <repository-url>
cd cem-rag

# 2. 创建虚拟环境
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装项目（开发模式）
pip install -e .

# 4. 安装评估依赖（可选）
pip install -e ".[eval]"
```

#### 2.3 验证安装

```bash
# 检查 zpkg 模块
python -c "from src.zpkg import builder, reader; print('✓ zpkg module OK')"

# 检查评估模块
python -c "from src.eval import longembed_eval; print('✓ eval module OK')"
```

## 3 核心功能使用

#### 3.1 构建 zpkg 文件

从原始 JSONL 语料库构建压缩容器：

```bash
python -m src.zpkg.builder \
    corpus.jsonl \
    output.zpkg \
    --target-chunk-size 4096 \
    --compression-level 3 \
    --dict-size 65536
```

**参数说明**：
- `corpus.jsonl`：输入文件，每行一个 JSON 对象，包含 `_id` 和 `text` 字段
- `output.zpkg`：输出的 zpkg 文件路径
- `--target-chunk-size`：目标压缩 chunk 大小（默认：4096 字节）
- `--compression-level`：zstd 压缩级别（1-22，默认：3）
- `--dict-size`：共享字典大小（默认：64KB）

**输出**：
- `output.zpkg`：压缩容器文件
- `output.zpkg.mapping.pkl`：corpus-id 到 chunk 索引的映射文件

#### 3.2 训练 CEM 模型

```bash
# 训练 CEM 模型（待实现）
python -m src.train.train \
    --zpkg-path data/corpus.zpkg \
    --teacher-model Qwen/Qwen3-Embedding-0.6B \
    --output-dir output/cem_model \
    --batch-size 32 \
    --epochs 10
```

#### 3.3 运行评估

**LongEmbed 评估**：

```bash
# 评估文本编码器
python src/eval/longembed_eval.py \
    --model Qwen/Qwen3-Embedding-0.6B \
    --output-dir results/qwen_0.6b \
    --device cuda \
    --batch-size 32

# 评估 CEM 编码器（待实现）
python src/eval/longembed_eval.py \
    --model cem \
    --cem-model-path output/cem_model \
    --zpkg-path data/corpus.zpkg \
    --output-dir results/cem
```

**BEIR 评估**（待实现）：

```bash
python src/eval/beir_eval.py \
    --model cem \
    --cem-model-path output/cem_model \
    --datasets msmarco trec-covid nq
```

#### 3.4 RAG 应用

```bash
# 构建索引（待实现）
python -m src.rag.indexing \
    --zpkg-path data/corpus.zpkg \
    --cem-model-path output/cem_model \
    --index-path output/faiss_index

# 运行检索（待实现）
python -m src.rag.retrieval \
    --index-path output/faiss_index \
    --query "your query here" \
    --top-k 10
```

#### 3.5 常见任务示例

**示例1：完整工作流**

```bash
# 1. 下载 BEIR 数据集
python scripts/download_beir.py

# 2. 构建 zpkg
python -m src.zpkg.builder data/msmarco/corpus.jsonl data/msmarco.zpkg

# 3. 训练 CEM（待实现）
python -m src.train.train --zpkg-path data/msmarco.zpkg ...

# 4. 评估
python src/eval/longembed_eval.py --model cem ...

# 5. RAG 应用（待实现）
python -m src.rag.indexing ...
```

**示例2：读取 zpkg 文件**

```python
from src.zpkg.reader import ZPKGReader

# 打开 zpkg 文件
reader = ZPKGReader("data/corpus.zpkg", mapping_path="data/corpus.zpkg.mapping.pkl")

# 获取压缩 chunk（用于 CEM 训练）
compressed_bytes = reader.get_compressed_chunk(chunk_index=0)

# 获取解压后的文档（用于 late decompression）
document = reader.get_document_by_corpus_id(corpus_id="doc_123")

# 获取统计信息
stats = reader.get_statistics()
print(f"Total chunks: {stats['num_chunks']}")
print(f"Compression ratio: {stats['compression_ratio']:.2f}")
```

## 4 常见问题（FAQ）

**Q: zpkg 文件可以跨平台使用吗？**  
A: 是的，zpkg 是二进制格式，跨平台兼容。

**Q: 如何查看 zpkg 文件内容？**  
A: 使用 `ZPKGReader` 的 `get_statistics()` 方法查看统计信息。

**Q: 训练 CEM 需要多少 GPU 内存？**  
A: 取决于 batch size 和模型大小，建议至少 16GB GPU 内存。

**Q: 支持哪些评估基准？**  
A: 目前支持 LongEmbed，BEIR 支持正在开发中。

---

## 5 整体架构与设计原则

项目采用**模块化架构**，包含 5 个核心模块，遵循清晰的依赖关系和层次结构：

**模块层次**：
- **层次1（基础设施层）**：`zpkg/` - 压缩容器，不依赖其他业务模块
- **层次2（编码层）**：`encoders/` - 依赖 `zpkg/`
- **层次3（应用层）**：`train/`, `eval/`, `rag/` - 依赖下层模块

**5大核心模块**：
1. **zpkg/** - 压缩容器模块：提供压缩数据的随机访问容器
2. **encoders/** - 编码器模块：提供统一的编码接口（MTEB兼容）
3. **train/** - 训练模块：CEM 模型训练
4. **eval/** - 评估模块：模型性能评估
5. **rag/** - RAG应用模块：端到端 RAG 系统实现

#### 5.1 全局不变量（Hard Constraints）

以下约束在任何实现、任何优化中都 **必须成立**：

**压缩域不变量**：
- **MUST NOT** 在索引、检索、评估阶段解压文档
- **MUST** 使用压缩字节流作为 CEM 的输入
- 解压 **ONLY ALLOWED** 在 Top-K 召回之后

**解耦原则**：
- zpkg **不感知** corpus-id、BEIR / RAG / 训练 / 评估
- 语义映射在 zpkg 构建时生成（`.mapping.pkl`），由 zpkg.reader 提供访问接口

**Query 原则（已冻结）**：
- Query 天生是用户输入文本
- **Query Encoder MUST be text-based**
- 使用 CEM 编码 query 被明确视为 **Non-Goal**
  - 原因：对 query 再压缩一次在语义与工程上均无意义

#### 5.2 明确 Non-Goals

- ❌ query 压缩
- ❌ zpkg 存语义信息
- ❌ 索引阶段解压
- ❌ end-to-end 生成式训练

#### 5.3 模块设计原则

1. **单一职责**：每个模块只负责一个明确的领域
2. **依赖方向**：只能依赖下层模块，不能循环依赖
3. **接口统一**：编码器模块提供统一接口，评估和RAG模块都基于此接口
4. **解耦设计**：模块间通过明确的接口交互，避免紧耦合
5. **可扩展性**：新模型、新评估任务、新训练方法都可以在对应模块中扩展

## 6 Zpkg 压缩容器模块

**职责**：提供压缩数据的随机访问容器

**核心功能**：
- zpkg 文件构建（`builder.py`）
- 压缩/解压缩 chunk 读取（`reader.py`）
- O(1) 随机访问任意压缩 chunk

**二进制布局**：

```
[ Header (32 bytes) ]
→ [ Dictionary Section (zstd shared dict) ]
→ [ Chunk 0 ]
→ [ Chunk 1 ]
→ ...
→ [ Chunk N-1 ]
→ [ Index Table (uint32 offsets) ]
```

**关键设计**：
- Header 中包含 `index_table_offset` 字段（8字节），指向 Index Table 的位置
- Index Table 在文件末尾，包含所有 chunk 的偏移量（uint32数组）
- 偏移量是相对于 chunks 起始位置的相对偏移
- chunk_i 的长度 = offset[i+1] - offset[i]
- 最后一个 chunk 的长度 = (index_table_offset - chunks_start) - offset[N-1]

**API**：

```python
# 构建
from src.zpkg.builder import ZPKGBuilder
builder = ZPKGBuilder(target_chunk_size=4096)
manifest = builder.build_zpkg("corpus.jsonl", "output.zpkg")

# 读取
from src.zpkg.reader import ZPKGReader
reader = ZPKGReader("output.zpkg", "output.zpkg.mapping.pkl")
compressed = reader.get_compressed_chunk(chunk_index=0)
document = reader.get_document_by_corpus_id(corpus_id="doc_123")
```

**设计原则**：
- zpkg **不感知** corpus-id、BEIR、RAG、训练、评估
- 语义映射在构建时生成（`.mapping.pkl` 文件），由 reader 提供访问接口
- 与任务无关的通用压缩容器

## 7 Encoders 编码器模块

**职责**：提供统一的编码接口（MTEB兼容）

**核心功能**：
- Text Encoders（Qwen, E5等 Teacher 模型）
- Compression Encoder（CEM，Student 模型）
- 统一 `encode()`, `encode_queries()`, `encode_corpus()` 接口
- Teacher embedding cache 管理（作为 encoder 的能力）

**模块结构**：

```
encoders/
├── __init__.py
├── base_embedding.py       # BaseEncoder（MTEB兼容接口）
├── text_encoders/          # 文本编码器
│   ├── qwen_embedding.py
│   └── e5_mistral_embedding.py
├── compression_encoder.py  # CEM编码器（基于BLT，待实现）
└── cache.py                # Teacher embedding缓存（可选）
```

**统一接口**：

```python
class BaseEmbeddingModel:
    def encode(self, texts: List[str]) -> np.ndarray:
        """编码文本列表"""
        pass
    
    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """编码查询列表"""
        pass
    
    def encode_corpus(self, corpus: List[str]) -> np.ndarray:
        """编码文档列表"""
        pass
```

**关键设计**：
- Teacher Cache：磁盘级缓存（npy / memmap），作为 encoder 的能力，避免重复计算 Teacher embedding
- 编码器提供统一接口，支持缓存机制

**依赖**：`zpkg/`（CEM需要读取压缩数据）  
**被使用**：`train/`, `eval/`, `rag/`

## 8 Train 训练模块

**职责**：CEM 模型训练

**核心功能**：
- 训练脚本（`train.py`）
- Loss 函数（Distillation, Contrastive, Triplet）
- Trainer 类
- 训练数据处理（Dataset、DataLoader、负样本采样）

**模块结构**（规划中）：

```
train/
├── __init__.py
├── train.py           # 主训练脚本
├── trainer.py         # Trainer类
├── losses.py          # Loss函数
├── dataset.py         # 训练Dataset（从zpkg读取，调用teacher cache）
└── model.py           # CEM模型定义（BLT wrapper + projection）
```

**关键设计**：
- 训练数据处理紧耦合在训练模块内，包括 Dataset、DataLoader、负样本采样策略
- 使用 `zpkg.reader` 读取压缩数据，使用 `encoders` 的 teacher cache 获取 embedding

**CEM 模型设计**：

- **Student**：BLT（Byte Latent Transformer）
  - 基于：https://huggingface.co/itazap/blt-1b-hf
  - 使用路径：byte-level encoder → local encoder → global transformer
  - **明确移除 / bypass decode 与预测路径**
  - 使用 wrapper，而非大幅 fork BLT

- **Teacher**：Qwen-0.6B / E5
  - 输入：解压后的原始文本
  - Embedding：last_hidden_state + attention-mask mean pooling
  - 输出维度：**1024**

- **Loss 设计**：
  ```
  L = α·L_distill + β·L_contrastive + γ·L_triplet
  ```
  - Distillation Loss：cosine / MSE on normalized embeddings
  - Contrastive Loss（InfoNCE）：正样本 (student_i, teacher_i)，负样本 batch 内其它样本
  - Triplet Loss：anchor=student_i, positive=teacher_i, negative=teacher_j

- **Teacher Cache**：Teacher embedding SHOULD be cached（磁盘级 cache：npy / memmap）

**依赖**：`encoders/`, `zpkg/`  
**被使用**：无（独立模块）

## 9 Eval 评估模块

**职责**：模型性能评估

**核心功能**：
- LongEmbed 评估
- BEIR 评估（规划中）
- 模态无关的评估框架
- 支持文本和压缩编码器的统一评估

**模块结构**：

```
eval/
├── __init__.py
├── longembed_eval.py  # LongEmbed评估（已实现）
└── beir_eval.py       # BEIR评估（规划中）
```

**设计哲学**：

> **Evaluator MUST be modality-agnostic**

评估器：
- 不区分文本 / 压缩
- 只认 corpus-id
- 统一的 encoder 接口

**LongEmbed Benchmark**：

包含 6 个任务：
- **2 个合成任务**：Passkey, Needle
- **4 个真实任务**：NarrativeQA, QMSum, SummScreenFD, 2WikiMultihopQA

**依赖**：`encoders/`, `zpkg/`  
**被使用**：无（独立模块）

## 10 RAG 应用模块

**职责**：端到端 RAG 系统实现

**核心功能**：
- 索引构建（使用 CEM 从压缩域生成 embedding）
- 检索流程（ANN search）
- Late decompression（Top-K 召回后解压）
- LangChain 集成
- LLM 生成

**模块结构**（规划中）：

```
rag/
├── __init__.py
├── indexing.py        # 索引构建（使用CEM）
├── vectorstore.py     # 向量库管理（pgvector封装）
├── retrieval.py       # 检索流程
└── chain.py           # LangChain集成
```

**索引流程**：

```
corpus-id
 → compressed bytes (zpkg)
 → CEM embedding
 → PostgreSQL VectorStore (pgvector)
```

**检索流程**：

```
query (text)
 → query encoder (text-based)
 → ANN search (pgvector)
 → Top-K corpus-id
 → 局部解压（仅解压召回的文档）
 → LangChain prompt
 → LLM
```

**关键约束**：
- **MUST NOT** 使用 `add_texts()`（传统 LangChain 方法）
- **MUST** 使用 CEM embedding 构建索引
- Query Encoder **MUST be text-based**（不压缩 query）
- 向量存储使用 **pgvector**（支持最低 PostgreSQL 12），而非 FAISS

**依赖**：`encoders/`, `zpkg/`  
**被使用**：无（独立模块）

---

## 11 开发环境设置

#### 11.1 项目结构

```
cem-rag/
├── src/                    # 源代码模块
│   ├── zpkg/              # 压缩容器模块（基础设施层）
│   ├── encoders/          # 编码器模块（统一接口层）
│   ├── train/             # 训练模块
│   ├── eval/              # 评估模块
│   ├── rag/               # RAG应用模块
│   └── common/            # 共享工具（可选）
├── tests/                 # 测试脚本
├── data/                  # 数据文件（gitignored）
├── models/                # 模型文件（HuggingFace缓存，gitignored）
├── output/                # 输出文件（结果、日志等，gitignored）
├── scripts/               # 工具脚本（下载、预处理等）
├── configs/               # 配置文件
├── README.md              # 主文档
├── requirements.txt       # Python 依赖
└── setup.py               # 包配置
```

#### 11.2 标准安装方式（推荐）

```bash
# 1. 创建虚拟环境（推荐使用 Python 3.11）
python3.11 -m venv venv
source venv/bin/activate  # 或 Windows: venv\Scripts\activate

# 2. 安装项目（开发模式）
pip install -e .

# 3. 安装评估依赖（如果需要运行 LongEmbed 评估）
pip install -e ".[eval]"

# 4. 安装开发工具（可选）
pip install -e ".[dev]"
```

#### 11.3 运行测试

请参考 `tests/README.md`。

#### 11.4 导入规范

- **src/ 内部**：使用相对导入或绝对导入（如 `from zpkg.reader import ZPKGReader`）
- **从 tests/ 或 scripts/**：脚本会自动添加 `src/` 到 `sys.path`，使用 `from zpkg.reader import ZPKGReader`
- **安装后（`pip install -e .`）**：所有导入在任何地方都正常工作

#### 11.5 添加新模块

1. 在 `src/` 中创建模块目录（如 `src/new_module/`）
2. 添加 `__init__.py` 使其成为包
3. 如需要，更新 `setup.py`
4. 在 `tests/` 中添加测试
5. 更新本 README 的模块说明

## 12 AI 代理开发指南

#### 12.1 理解项目架构

**关键原则**：
- 本项目是**设计规范驱动**的，README 是最高优先级参考
- 5 大模块有明确的依赖关系和职责划分
- 必须遵守全局不变量（Hard Constraints）

**开发前必读**：
1. 第一部分：理解项目目的和研究意义
2. 第三部分：理解 5 大模块的职责和依赖关系
3. 第三部分 5.1：理解全局不变量，**不得违反**

#### 12.2 常见陷阱与避免方法

**陷阱1：在索引/检索阶段解压文档**

❌ **错误**：
```python
# 错误：在索引阶段解压
text = reader.get_decompressed_chunk(chunk_index)
embedding = text_encoder.encode([text])
```

✅ **正确**：
```python
# 正确：使用压缩字节
compressed_bytes = reader.get_compressed_chunk(chunk_index)
embedding = cem_encoder.encode([compressed_bytes])
```

**陷阱2：让 zpkg 感知业务逻辑**

❌ **错误**：
```python
# 错误：在 zpkg 模块中添加 corpus-id 相关代码
class ZPKGReader:
    def get_by_corpus_id(self, corpus_id):  # 不应该在这里
        ...
```

✅ **正确**：
```python
# 正确：语义映射外挂
reader = ZPKGReader("file.zpkg")
mapping = load_mapping("file.zpkg.mapping.pkl")
chunk_indices = mapping[corpus_id]
chunks = [reader.get_compressed_chunk(i) for i in chunk_indices]
```

**陷阱3：使用 CEM 编码 query**

❌ **错误**：
```python
# 错误：压缩 query
compressed_query = compress(query_text)
query_embedding = cem_encoder.encode([compressed_query])
```

✅ **正确**：
```python
# 正确：query 始终使用文本编码器
query_embedding = text_encoder.encode_queries([query_text])
```

**陷阱4：忽略 Teacher Cache**

❌ **错误**：
```python
# 错误：每次都重新计算 Teacher embedding
for batch in dataloader:
    teacher_embeddings = teacher_model.encode(batch['texts'])  # 慢！
```

✅ **正确**：
```python
# 正确：使用 Teacher Cache
teacher_cache = TeacherCache(cache_path="teacher_embeddings.npy")
for batch in dataloader:
    teacher_embeddings = teacher_cache.get_or_compute(batch['corpus_ids'])
```

**陷阱5：破坏模块依赖关系**

❌ **错误**：
```python
# 错误：违反依赖方向
from src.train import Trainer  # 上层模块依赖下层模块！
```

✅ **正确**：
```python
# 正确：遵循依赖方向
from src.zpkg import ZPKGReader  # 依赖方向正确
```

#### 12.3 实现新功能的标准流程

**步骤1：确定模块归属**

根据功能确定应该放在哪个模块：
- 压缩相关 → `zpkg/`
- 编码逻辑 → `encoders/`
- 训练逻辑 → `train/`
- 评估逻辑 → `eval/`
- RAG 应用 → `rag/`

**步骤2：检查依赖关系**

确保：
- 只依赖下层模块
- 不创建循环依赖
- 遵循单一职责原则

**步骤3：实现统一接口**

如果是编码器，必须实现：
- `encode()`
- `encode_queries()`（可选，默认调用 encode）
- `encode_corpus()`（可选，默认调用 encode）

**步骤4：遵守全局不变量**

检查是否违反：
- 压缩域不变量
- 解耦原则
- Query 原则

**步骤5：添加测试**

在 `tests/` 中添加测试，确保功能正确。

#### 13.4 代码风格建议

- **命名**：使用清晰的命名，避免缩写
- **文档**：关键函数和类必须有 docstring
- **类型提示**：使用 Python 类型提示
- **错误处理**：提供清晰的错误信息

#### 12.5 调试技巧

**检查 zpkg 文件**：
```python
from src.zpkg.reader import ZPKGReader
reader = ZPKGReader("file.zpkg")
stats = reader.get_statistics()
print(stats)
```

**检查模块依赖**：
```python
# 检查导入是否违反依赖关系
import sys
print(sys.modules.keys())
```

**检查编码器接口**：
```python
from src.encoders.base_embedding import BaseEmbeddingModel
# 确保新编码器继承 BaseEmbeddingModel
```

---

## 13 参考资源

#### 13.1 论文

- **BLT**：[Byte Latent Transformer: Patches Scale Better Than Tokens](https://arxiv.org/abs/2412.09871)
- **E5**：[Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/abs/2212.03533)
- **LongEmbed**：[LongEmbed: Extending Embedding Models for Long Context Retrieval](https://arxiv.org/abs/2404.12096)
- **Qwen3 Embedding**：[Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models](https://arxiv.org/abs/2506.05176)

#### 13.2 模型

- **E5-Mistral-7B**：https://huggingface.co/intfloat/e5-mistral-7b-instruct
- **Qwen3-Embedding-0.6B**：https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- **Qwen3-Embedding-4B**：https://huggingface.co/Qwen/Qwen3-Embedding-4B
- **Qwen3-Embedding-8B**：https://huggingface.co/Qwen/Qwen3-Embedding-8B

#### 13.3 仓库

- **BLT**：https://github.com/facebookresearch/blt

#### 13.4 基准测试

- **MTEB Benchmark**：https://huggingface.co/spaces/mteb/leaderboard
