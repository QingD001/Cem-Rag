#!/usr/bin/env python3
"""
测试随机解压功能：验证映射表是否正确
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

import pickle
import json
import random
from zpkg.reader import ZPKGReader

# 文件路径
zpkg_path = "output/trec-covid_cli.zpkg"
mapping_path = "output/trec-covid_cli.zpkg.mapping.pkl"
original_path = "data/trec-covid/trec-covid/corpus.jsonl"

# 加载映射表
print("Loading mapping table...")
with open(mapping_path, 'rb') as f:
    corpus_to_chunks = pickle.load(f)
print(f"Loaded mapping for {len(corpus_to_chunks):,} corpus IDs")

# 统计多chunk的文档数量
multi_chunk_ids = [cid for cid, chunks in corpus_to_chunks.items() if len(chunks) > 1]
single_chunk_ids = [cid for cid, chunks in corpus_to_chunks.items() if len(chunks) == 1]
print(f"Single-chunk documents: {len(single_chunk_ids):,} ({len(single_chunk_ids)/len(corpus_to_chunks)*100:.1f}%)")
print(f"Multi-chunk documents: {len(multi_chunk_ids):,} ({len(multi_chunk_ids)/len(corpus_to_chunks)*100:.1f}%)")

# 加载原始文档（只加载测试需要的）
test_size = 100
test_ids = []
if single_chunk_ids:
    test_ids.extend(random.sample(single_chunk_ids, min(80, len(single_chunk_ids))))
if multi_chunk_ids:
    test_ids.extend(random.sample(multi_chunk_ids, min(20, len(multi_chunk_ids))))

print(f"\nLoading {len(test_ids)} test documents...")
original_docs = {}
with open(original_path, 'r') as f:
    for line in f:
        doc = json.loads(line)
        corpus_id = doc.get('_id', '')
        if corpus_id in test_ids:
            original_docs[corpus_id] = doc.get('text', '') + '\n'
            if len(original_docs) >= len(test_ids):
                break

print(f"Loaded {len(original_docs)} test documents")
print(f"Testing {len(test_ids)} corpus IDs ({len([cid for cid in test_ids if cid in single_chunk_ids])} single-chunk, {len([cid for cid in test_ids if cid in multi_chunk_ids])} multi-chunk):")
print("=" * 70)

# 打开zpkg文件
success_count = 0
fail_count = 0
failed_ids = []
multi_chunk_success = 0
multi_chunk_fail = 0

with ZPKGReader(zpkg_path) as reader:
    for i, corpus_id in enumerate(test_ids, 1):
        if corpus_id not in original_docs:
            continue
        
        original_text = original_docs[corpus_id]
        chunk_indices = corpus_to_chunks[corpus_id]
        is_multi_chunk = len(chunk_indices) > 1
        
        # 从所有相关的chunks中还原文档
        # Use get_decompressed_chunks to properly handle UTF-8 boundaries
        try:
            if len(chunk_indices) == 1:
                combined_text = reader.get_decompressed_chunk(chunk_indices[0])
            else:
                combined_text = reader.get_decompressed_chunks(chunk_indices)
        except Exception as e:
            print(f"[{i}] {corpus_id}: ERROR reading chunks: {e}")
            fail_count += 1
            if is_multi_chunk:
                multi_chunk_fail += 1
            failed_ids.append(corpus_id)
            continue
        
        # 检查原始文档是否在合并后的文本中
        if original_text in combined_text:
            success_count += 1
            if is_multi_chunk:
                multi_chunk_success += 1
            if i <= 10:  # 显示前10个的详细信息
                print(f"[{i}] {corpus_id}: ✓ SUCCESS ({len(chunk_indices)} chunk(s))")
        else:
            fail_count += 1
            if is_multi_chunk:
                multi_chunk_fail += 1
            failed_ids.append(corpus_id)
            if i <= 10 or is_multi_chunk:  # 显示前10个或多chunk的详细信息
                print(f"[{i}] {corpus_id}: ✗ FAILED ({len(chunk_indices)} chunk(s))")
                # 尝试找部分匹配
                if original_text[:100] in combined_text:
                    pos = combined_text.find(original_text[:100])
                    extracted = combined_text[pos:pos+len(original_text)]
                    print(f"      Found first 100 chars at position {pos}, but length mismatch: {len(extracted)} vs {len(original_text)}")
        
        if i % 20 == 0:
            print(f"Progress: {i}/{len(test_ids)} tested, {success_count} success, {fail_count} failed")

print("\n" + "=" * 70)
print(f"Test Summary:")
print(f"  Total tested: {len(test_ids)}")
print(f"  Success: {success_count} ({success_count/len(test_ids)*100:.1f}%)")
print(f"  Failed: {fail_count} ({fail_count/len(test_ids)*100:.1f}%)")
if multi_chunk_ids:
    print(f"\n  Multi-chunk documents:")
    print(f"    Success: {multi_chunk_success} ({multi_chunk_success/(multi_chunk_success+multi_chunk_fail)*100:.1f}%)")
    print(f"    Failed: {multi_chunk_fail}")
if failed_ids:
    print(f"\nFailed corpus IDs (first 10): {failed_ids[:10]}")

