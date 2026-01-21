# 测试说明

本目录包含项目的pytest单元测试。

## 运行测试

### 运行所有测试
```bash
pytest
```

### 运行特定测试文件
```bash
pytest tests/test_zpkg.py
```

### 运行特定测试函数
```bash
pytest tests/test_zpkg.py::test_single_chunk_mappings_contain_content
pytest tests/test_zpkg.py::test_single_chunk_decompression
```

### 运行特定参数化的测试
```bash
pytest tests/test_zpkg.py -k "single_chunk"
pytest tests/test_zpkg.py -k "mapping"
pytest tests/test_zpkg.py -k "decompress"
```

### 显示详细输出
```bash
pytest -v
pytest -vv  # 更详细
```

### 显示print输出
```bash
pytest -s
```

## 测试结构

### test_zpkg.py
zpkg压缩容器的所有单元测试，包含以下测试函数：

**映射精确性验证**：
- `test_single_chunk_mappings_contain_content`: 验证单chunk映射包含corpus内容
- `test_multi_chunk_mappings_contain_content`: 验证多chunk映射包含corpus内容
- `test_mapped_chunk_indices_are_valid`: 验证映射的chunk索引有效
- `test_all_corpus_ids_exist_in_original`: 验证所有corpus ID存在于原始corpus
- `test_no_empty_mappings`: 验证没有空映射

**随机解压功能**：
- `test_single_chunk_decompression`: 测试单chunk文档解压
- `test_multi_chunk_decompression`: 测试多chunk文档解压
- `test_decompressed_content_matches_original`: 测试解压内容匹配原始corpus
- `test_utf8_boundary_handling`: 测试UTF-8边界处理

## 测试数据要求

测试需要以下数据文件：
- `output/*.zpkg`: zpkg压缩文件
- `output/*.mapping.pkl`: 映射文件
- `data/nq.jsonl` 或 `data/trec-covid/trec-covid/corpus.jsonl`: 原始corpus文件

如果数据文件不存在，相关测试会被自动跳过。

## 测试原则

每个测试函数遵循单一职责原则，只测试一件事：
- 每个测试函数有明确的测试目标
- 使用assert进行断言，而不是print
- 使用pytest的fixtures进行测试数据准备
- 使用parametrize进行参数化测试
