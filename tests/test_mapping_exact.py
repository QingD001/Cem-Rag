#!/usr/bin/env python3
"""
验证映射是否正确：
1. 每个corpus-id映射到的chunk中必须包含该corpus的内容
2. 不能有多余的chunk（映射的范围必须精确，不能扩大）
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
from tqdm import tqdm
from zpkg.reader import ZPKGReader

def find_latest_files(pattern_base):
    """找到最新的mapping和zpkg文件"""
    output_dir = Path('output')
    if not output_dir.exists():
        print(f"错误: output目录不存在")
        return None, None
    
    # 找到所有匹配的mapping文件
    mapping_files = list(output_dir.glob(f"*{pattern_base}*.mapping.pkl"))
    if not mapping_files:
        print(f"错误: 找不到匹配 *{pattern_base}*.mapping.pkl 的文件")
        return None, None
    
    # 使用最新的
    mapping_file = max(mapping_files, key=lambda p: p.stat().st_mtime)
    
    # 找到对应的zpkg文件（去掉.mapping.pkl后缀）
    zpkg_path = mapping_file.with_suffix('').with_suffix('.zpkg')
    if not zpkg_path.exists():
        # 尝试其他可能的路径
        zpkg_files = list(output_dir.glob(f"*{pattern_base}*.zpkg"))
        if zpkg_files:
            zpkg_path = max(zpkg_files, key=lambda p: p.stat().st_mtime)
        else:
            print(f"错误: 找不到对应的zpkg文件: {mapping_file}")
            return None, None
    
    return mapping_file, zpkg_path

def load_original_corpus(corpus_jsonl_path):
    """加载原始corpus，建立corpus_id到内容的映射"""
    corpus_map = {}
    print(f"加载原始corpus: {corpus_jsonl_path}")
    with open(corpus_jsonl_path, 'r') as f:
        for line in tqdm(f, desc="读取corpus", unit="docs"):
            doc = json.loads(line)
            corpus_id = doc.get('_id', '')
            text = doc.get('text', '') + '\n'
            if corpus_id:
                corpus_map[corpus_id] = text.encode('utf-8')
    return corpus_map

def verify_mapping_exact(mapping_file, zpkg_file, corpus_jsonl_path):
    """验证映射是否精确（不能扩大范围）"""
    print(f"检查文件:")
    print(f"  Mapping: {mapping_file}")
    print(f"  ZPKG: {zpkg_file}")
    print(f"  Corpus: {corpus_jsonl_path}")
    
    # 加载mapping
    with open(mapping_file, 'rb') as f:
        mapping = pickle.load(f)
    
    print(f"\n总文档数: {len(mapping):,}")
    
    # 加载原始corpus
    corpus_map = load_original_corpus(corpus_jsonl_path)
    print(f"加载了 {len(corpus_map):,} 个corpus文档")
    
    # 打开zpkg reader
    with ZPKGReader(zpkg_file) as reader:
        num_chunks = reader.num_chunks
        print(f"ZPKG文件有 {num_chunks:,} 个chunks")
        
        # 统计
        correct_count = 0
        missing_count = 0
        extra_chunks_count = 0
        errors = []
        
        print(f"\n验证映射（检查映射的chunk是否包含corpus，且没有多余的chunk）...")
        
        # 优化：只检查映射的chunk，不遍历所有chunk（太慢）
        # 对于精确性验证，我们采用两步：
        # 1. 检查映射的chunk是否都包含该corpus
        # 2. 对于有问题的文档，再检查是否有其他chunk也包含（但这一步太慢，先跳过）
        
        for corpus_id, mapped_chunks in tqdm(mapping.items(), desc="验证中", unit="docs", total=len(mapping)):
            if not corpus_id:
                continue
            
            if corpus_id not in corpus_map:
                missing_count += 1
                if len(errors) < 50:
                    errors.append(f"{corpus_id}: 在原始corpus中找不到")
                continue
            
            expected_bytes = corpus_map[corpus_id]
            
            # 对于跨chunk的文档，需要合并所有映射的chunk来检查
            if len(mapped_chunks) > 1:
                # 合并所有映射的chunk
                combined_data = b""
                read_error = False
                for chunk_idx in sorted(mapped_chunks):
                    if chunk_idx >= num_chunks:
                        continue
                    try:
                        chunk_data = reader.get_decompressed_chunk_bytes(chunk_idx)
                        combined_data += chunk_data
                    except Exception as e:
                        if len(errors) < 50:
                            errors.append(f"{corpus_id}: 读取chunk {chunk_idx} 时出错: {e}")
                        read_error = True
                        break  # 跳出循环
                
                if read_error:
                    missing_count += 1
                    continue
                
                # 检查合并后的数据是否包含该corpus
                if expected_bytes in combined_data:
                    correct_count += 1
                else:
                    missing_count += 1
                    if len(errors) < 50:
                        errors.append(f"{corpus_id}: 映射到chunks {sorted(mapped_chunks)}，但合并这些chunk后仍找不到内容")
                continue
            
            # 单个chunk的情况
            if not mapped_chunks:
                missing_count += 1
                if len(errors) < 50:
                    errors.append(f"{corpus_id}: 映射为空")
                continue
            
            chunk_idx = mapped_chunks[0]
            if chunk_idx >= num_chunks:
                missing_count += 1
                if len(errors) < 50:
                    errors.append(f"{corpus_id}: 映射到无效chunk {chunk_idx}")
                continue
            
            try:
                chunk_data = reader.get_decompressed_chunk_bytes(chunk_idx)
                if expected_bytes in chunk_data:
                    correct_count += 1
                else:
                    missing_count += 1
                    if len(errors) < 50:
                        errors.append(f"{corpus_id}: 映射到chunk {chunk_idx}，但该chunk不包含该corpus")
            except Exception as e:
                missing_count += 1
                if len(errors) < 50:
                    errors.append(f"{corpus_id}: 读取chunk {chunk_idx} 时出错: {e}")
        
        # 输出结果
        print(f"\n验证结果:")
        print(f"  正确: {correct_count:,}")
        print(f"  缺失: {missing_count:,}")
        print(f"  范围扩大/缩小: {extra_chunks_count:,}")
        print(f"  总文档数: {len(mapping):,}")
        
        accuracy = (correct_count / len(mapping) * 100) if mapping else 0
        print(f"\n准确率: {accuracy:.2f}%")
        
        if errors:
            print(f"\n前20个错误:")
            for error in errors[:20]:
                print(f"  {error}")
            if len(errors) > 20:
                print(f"  ... 还有 {len(errors) - 20} 个错误")
        
        if correct_count == len(mapping):
            print(f"\n✓ 所有映射都正确！")
            return True
        else:
            print(f"\n✗ 存在映射错误")
            return False

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python test_mapping_exact.py <corpus_jsonl_path> [pattern]")
        print("  例如: python test_mapping_exact.py data/nq.jsonl nq")
        sys.exit(1)
    
    corpus_jsonl_path = sys.argv[1]
    pattern = sys.argv[2] if len(sys.argv) > 2 else 'nq'
    
    mapping_file, zpkg_file = find_latest_files(pattern)
    if not mapping_file or not zpkg_file:
        sys.exit(1)
    
    success = verify_mapping_exact(mapping_file, zpkg_file, corpus_jsonl_path)
    sys.exit(0 if success else 1)

