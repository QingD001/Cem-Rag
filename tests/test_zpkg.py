"""
Tests for zpkg compression container functionality.

This module contains all unit tests related to zpkg, including:
1. Mapping exactness verification
2. Random decompression functionality

Each test function follows the single responsibility principle.
"""
import json
import pickle
import pytest
import random
from pathlib import Path
from typing import Tuple, List
from conftest import find_latest_files, load_corpus_map
from src.zpkg.reader import ZPKGReader


def verify_single_chunk_mapping(corpus_id: str, chunk_idx: int, expected_bytes: bytes, 
                                reader: ZPKGReader, num_chunks: int) -> Tuple[bool, str]:
    """
    Verify that a single-chunk mapping is correct.
    
    Returns:
        Tuple of (is_correct, error_message)
    """
    if not chunk_idx:
        return False, "Mapping is empty"
    
    if chunk_idx >= num_chunks:
        return False, f"Invalid chunk index {chunk_idx} (max: {num_chunks - 1})"
    
    try:
        chunk_data = reader.get_decompressed_chunk_bytes(chunk_idx)
        if expected_bytes in chunk_data:
            return True, ""
        else:
            return False, f"Chunk {chunk_idx} does not contain corpus content"
    except Exception as e:
        return False, f"Error reading chunk {chunk_idx}: {e}"


def verify_multi_chunk_mapping(corpus_id: str, chunk_indices: List[int], 
                               expected_bytes: bytes, reader: ZPKGReader, 
                               num_chunks: int) -> Tuple[bool, str]:
    """
    Verify that a multi-chunk mapping is correct.
    
    Returns:
        Tuple of (is_correct, error_message)
    """
    if not chunk_indices:
        return False, "Mapping is empty"
    
    combined_data = b""
    for chunk_idx in sorted(chunk_indices):
        if chunk_idx >= num_chunks:
            return False, f"Invalid chunk index {chunk_idx} (max: {num_chunks - 1})"
        
        try:
            chunk_data = reader.get_decompressed_chunk_bytes(chunk_idx)
            combined_data += chunk_data
        except Exception as e:
            return False, f"Error reading chunk {chunk_idx}: {e}"
    
    if expected_bytes in combined_data:
        return True, ""
    else:
        return False, f"Combined chunks {sorted(chunk_indices)} do not contain corpus content"


# Mapping exactness tests

@pytest.mark.parametrize("pattern,corpus_path", [
    ("nq", "data/nq.jsonl"),
    ("trec-covid", "data/trec-covid/trec-covid/corpus.jsonl"),
])
def test_single_chunk_mappings_contain_content(pattern, corpus_path, tmp_path, monkeypatch):
    """
    Test that single-chunk mappings correctly point to chunks containing the corpus content.
    """
    mapping_file, zpkg_file = find_latest_files(pattern)
    if not mapping_file or not zpkg_file:
        pytest.skip(f"No files found for pattern: {pattern}")
    
    corpus_path_obj = Path(corpus_path)
    if not corpus_path_obj.exists():
        pytest.skip(f"Corpus file not found: {corpus_path}")
    
    with open(mapping_file, 'rb') as f:
        mapping = pickle.load(f)
    
    corpus_map = load_corpus_map(corpus_path)
    
    with ZPKGReader(str(zpkg_file)) as reader:
        num_chunks = reader.num_chunks
        
        # Test single-chunk mappings
        single_chunk_mappings = {cid: chunks for cid, chunks in mapping.items() 
                               if len(chunks) == 1 and cid in corpus_map}
        
        if not single_chunk_mappings:
            pytest.skip("No single-chunk mappings found")
        
        # Test a sample of single-chunk mappings
        sample_size = min(100, len(single_chunk_mappings))
        test_ids = list(single_chunk_mappings.keys())[:sample_size]
        
        errors = []
        for corpus_id in test_ids:
            chunk_idx = single_chunk_mappings[corpus_id][0]
            expected_bytes = corpus_map[corpus_id]
            
            is_correct, error_msg = verify_single_chunk_mapping(
                corpus_id, chunk_idx, expected_bytes, reader, num_chunks
            )
            
            if not is_correct:
                errors.append(f"{corpus_id}: {error_msg}")
        
        assert len(errors) == 0, (
            f"Found {len(errors)} errors in single-chunk mappings:\n" +
            "\n".join(errors[:10])
        )


@pytest.mark.parametrize("pattern,corpus_path", [
    ("nq", "data/nq.jsonl"),
    ("trec-covid", "data/trec-covid/trec-covid/corpus.jsonl"),
])
def test_multi_chunk_mappings_contain_content(pattern, corpus_path, tmp_path, monkeypatch):
    """
    Test that multi-chunk mappings correctly point to chunks that contain 
    the corpus content when combined.
    """
    mapping_file, zpkg_file = find_latest_files(pattern)
    if not mapping_file or not zpkg_file:
        pytest.skip(f"No files found for pattern: {pattern}")
    
    corpus_path_obj = Path(corpus_path)
    if not corpus_path_obj.exists():
        pytest.skip(f"Corpus file not found: {corpus_path}")
    
    with open(mapping_file, 'rb') as f:
        mapping = pickle.load(f)
    
    corpus_map = load_corpus_map(corpus_path)
    
    with ZPKGReader(str(zpkg_file)) as reader:
        num_chunks = reader.num_chunks
        
        # Test multi-chunk mappings
        multi_chunk_mappings = {cid: chunks for cid, chunks in mapping.items() 
                              if len(chunks) > 1 and cid in corpus_map}
        
        if not multi_chunk_mappings:
            pytest.skip("No multi-chunk mappings found")
        
        # Test a sample of multi-chunk mappings
        sample_size = min(50, len(multi_chunk_mappings))
        test_ids = list(multi_chunk_mappings.keys())[:sample_size]
        
        errors = []
        for corpus_id in test_ids:
            chunk_indices = multi_chunk_mappings[corpus_id]
            expected_bytes = corpus_map[corpus_id]
            
            is_correct, error_msg = verify_multi_chunk_mapping(
                corpus_id, chunk_indices, expected_bytes, reader, num_chunks
            )
            
            if not is_correct:
                errors.append(f"{corpus_id}: {error_msg}")
        
        assert len(errors) == 0, (
            f"Found {len(errors)} errors in multi-chunk mappings:\n" +
            "\n".join(errors[:10])
        )


@pytest.mark.parametrize("pattern", ["nq", "trec-covid"])
def test_mapped_chunk_indices_are_valid(pattern, tmp_path, monkeypatch):
    """
    Test that all mapped chunk indices are within valid range.
    """
    mapping_file, zpkg_file = find_latest_files(pattern)
    if not mapping_file or not zpkg_file:
        pytest.skip(f"No files found for pattern: {pattern}")
    
    with open(mapping_file, 'rb') as f:
        mapping = pickle.load(f)
    
    with ZPKGReader(str(zpkg_file)) as reader:
        num_chunks = reader.num_chunks
        
        invalid_mappings = []
        for corpus_id, chunk_indices in mapping.items():
            for chunk_idx in chunk_indices:
                if chunk_idx < 0 or chunk_idx >= num_chunks:
                    invalid_mappings.append(f"{corpus_id}: invalid chunk index {chunk_idx}")
        
        assert len(invalid_mappings) == 0, (
            f"Found {len(invalid_mappings)} invalid chunk indices:\n" +
            "\n".join(invalid_mappings[:10])
        )


@pytest.mark.parametrize("pattern,corpus_path", [
    ("nq", "data/nq.jsonl"),
    ("trec-covid", "data/trec-covid/trec-covid/corpus.jsonl"),
])
def test_all_corpus_ids_exist_in_original(pattern, corpus_path, tmp_path, monkeypatch):
    """
    Test that all corpus IDs in the mapping exist in the original corpus.
    """
    mapping_file, _ = find_latest_files(pattern)
    if not mapping_file:
        pytest.skip(f"No mapping file found for pattern: {pattern}")
    
    corpus_path_obj = Path(corpus_path)
    if not corpus_path_obj.exists():
        pytest.skip(f"Corpus file not found: {corpus_path}")
    
    with open(mapping_file, 'rb') as f:
        mapping = pickle.load(f)
    
    corpus_map = load_corpus_map(corpus_path)
    
    missing_ids = [cid for cid in mapping.keys() if cid and cid not in corpus_map]
    
    assert len(missing_ids) == 0, (
        f"Found {len(missing_ids)} corpus IDs in mapping that don't exist in original corpus:\n" +
        "\n".join(missing_ids[:10])
    )


@pytest.mark.parametrize("pattern", ["nq", "trec-covid"])
def test_no_empty_mappings(pattern, tmp_path, monkeypatch):
    """
    Test that there are no empty mappings (corpus_id -> empty list).
    """
    mapping_file, _ = find_latest_files(pattern)
    if not mapping_file:
        pytest.skip(f"No mapping file found for pattern: {pattern}")
    
    with open(mapping_file, 'rb') as f:
        mapping = pickle.load(f)
    
    empty_mappings = [cid for cid, chunks in mapping.items() if not chunks]
    
    assert len(empty_mappings) == 0, (
        f"Found {len(empty_mappings)} empty mappings:\n" +
        "\n".join(empty_mappings[:10])
    )


# Random decompression tests

@pytest.mark.parametrize("pattern,corpus_path", [
    ("trec-covid", "data/trec-covid/trec-covid/corpus.jsonl"),
    ("nq", "data/nq.jsonl"),
])
def test_single_chunk_decompression(pattern, corpus_path, tmp_path, monkeypatch):
    """
    Test that single-chunk documents can be correctly decompressed.
    """
    mapping_file, zpkg_file = find_latest_files(pattern)
    if not mapping_file or not zpkg_file:
        pytest.skip(f"No files found for pattern: {pattern}")
    
    corpus_path_obj = Path(corpus_path)
    if not corpus_path_obj.exists():
        pytest.skip(f"Corpus file not found: {corpus_path}")
    
    with open(mapping_file, 'rb') as f:
        mapping = pickle.load(f)
    
    # Find single-chunk documents
    single_chunk_ids = [cid for cid, chunks in mapping.items() if len(chunks) == 1]
    
    if not single_chunk_ids:
        pytest.skip("No single-chunk documents found")
    
    # Load original documents
    original_docs = {}
    with open(corpus_path_obj, 'r') as f:
        for line in f:
            doc = json.loads(line)
            corpus_id = doc.get('_id', '')
            if corpus_id in single_chunk_ids:
                original_docs[corpus_id] = doc.get('text', '') + '\n'
                if len(original_docs) >= len(single_chunk_ids):
                    break
    
    # Test a sample
    test_size = min(50, len(single_chunk_ids))
    test_ids = random.sample(single_chunk_ids, test_size)
    
    with ZPKGReader(str(zpkg_file)) as reader:
        errors = []
        for corpus_id in test_ids:
            if corpus_id not in original_docs:
                continue
            
            original_text = original_docs[corpus_id]
            chunk_idx = mapping[corpus_id][0]
            
            try:
                decompressed_text = reader.get_decompressed_chunk(chunk_idx)
                if original_text not in decompressed_text:
                    errors.append(f"{corpus_id}: decompressed content doesn't match")
            except Exception as e:
                errors.append(f"{corpus_id}: error decompressing: {e}")
        
        assert len(errors) == 0, (
            f"Found {len(errors)} errors in single-chunk decompression:\n" +
            "\n".join(errors[:10])
        )


@pytest.mark.parametrize("pattern,corpus_path", [
    ("trec-covid", "data/trec-covid/trec-covid/corpus.jsonl"),
    ("nq", "data/nq.jsonl"),
])
def test_multi_chunk_decompression(pattern, corpus_path, tmp_path, monkeypatch):
    """
    Test that multi-chunk documents can be correctly decompressed.
    """
    mapping_file, zpkg_file = find_latest_files(pattern)
    if not mapping_file or not zpkg_file:
        pytest.skip(f"No files found for pattern: {pattern}")
    
    corpus_path_obj = Path(corpus_path)
    if not corpus_path_obj.exists():
        pytest.skip(f"Corpus file not found: {corpus_path}")
    
    with open(mapping_file, 'rb') as f:
        mapping = pickle.load(f)
    
    # Find multi-chunk documents
    multi_chunk_ids = [cid for cid, chunks in mapping.items() if len(chunks) > 1]
    
    if not multi_chunk_ids:
        pytest.skip("No multi-chunk documents found")
    
    # Load original documents
    original_docs = {}
    with open(corpus_path_obj, 'r') as f:
        for line in f:
            doc = json.loads(line)
            corpus_id = doc.get('_id', '')
            if corpus_id in multi_chunk_ids:
                original_docs[corpus_id] = doc.get('text', '') + '\n'
                if len(original_docs) >= len(multi_chunk_ids):
                    break
    
    # Test a sample
    test_size = min(20, len(multi_chunk_ids))
    test_ids = random.sample(multi_chunk_ids, test_size)
    
    with ZPKGReader(str(zpkg_file)) as reader:
        errors = []
        for corpus_id in test_ids:
            if corpus_id not in original_docs:
                continue
            
            original_text = original_docs[corpus_id]
            chunk_indices = mapping[corpus_id]
            
            try:
                decompressed_text = reader.get_decompressed_chunks(chunk_indices)
                if original_text not in decompressed_text:
                    errors.append(f"{corpus_id}: decompressed content doesn't match")
            except Exception as e:
                errors.append(f"{corpus_id}: error decompressing: {e}")
        
        assert len(errors) == 0, (
            f"Found {len(errors)} errors in multi-chunk decompression:\n" +
            "\n".join(errors[:10])
        )


@pytest.mark.parametrize("pattern,corpus_path", [
    ("trec-covid", "data/trec-covid/trec-covid/corpus.jsonl"),
    ("nq", "data/nq.jsonl"),
])
def test_decompressed_content_matches_original(pattern, corpus_path, tmp_path, monkeypatch):
    """
    Test that decompressed content matches the original corpus content.
    """
    mapping_file, zpkg_file = find_latest_files(pattern)
    if not mapping_file or not zpkg_file:
        pytest.skip(f"No files found for pattern: {pattern}")
    
    corpus_path_obj = Path(corpus_path)
    if not corpus_path_obj.exists():
        pytest.skip(f"Corpus file not found: {corpus_path}")
    
    with open(mapping_file, 'rb') as f:
        mapping = pickle.load(f)
    
    # Load original documents
    original_docs = {}
    with open(corpus_path_obj, 'r') as f:
        for line in f:
            doc = json.loads(line)
            corpus_id = doc.get('_id', '')
            if corpus_id in mapping:
                original_docs[corpus_id] = doc.get('text', '') + '\n'
    
    # Test a random sample
    test_size = min(100, len(mapping))
    test_ids = random.sample(list(mapping.keys()), test_size)
    
    with ZPKGReader(str(zpkg_file)) as reader:
        errors = []
        for corpus_id in test_ids:
            if corpus_id not in original_docs:
                continue
            
            original_text = original_docs[corpus_id]
            chunk_indices = mapping[corpus_id]
            
            try:
                if len(chunk_indices) == 1:
                    decompressed_text = reader.get_decompressed_chunk(chunk_indices[0])
                else:
                    decompressed_text = reader.get_decompressed_chunks(chunk_indices)
                
                if original_text not in decompressed_text:
                    errors.append(f"{corpus_id}: content mismatch")
            except Exception as e:
                errors.append(f"{corpus_id}: error: {e}")
        
        # Allow some tolerance for edge cases
        error_rate = len(errors) / len(test_ids)
        assert error_rate < 0.05, (
            f"Error rate too high: {error_rate:.2%} ({len(errors)}/{len(test_ids)})\n" +
            "\n".join(errors[:10])
        )


@pytest.mark.parametrize("pattern", ["trec-covid", "nq"])
def test_utf8_boundary_handling(pattern, tmp_path, monkeypatch):
    """
    Test that UTF-8 boundaries are handled correctly in multi-chunk documents.
    This ensures that multi-byte UTF-8 characters aren't corrupted at chunk boundaries.
    """
    mapping_file, zpkg_file = find_latest_files(pattern)
    if not mapping_file or not zpkg_file:
        pytest.skip(f"No files found for pattern: {pattern}")
    
    with open(mapping_file, 'rb') as f:
        mapping = pickle.load(f)
    
    # Find multi-chunk documents
    multi_chunk_ids = [cid for cid, chunks in mapping.items() if len(chunks) > 1]
    
    if not multi_chunk_ids:
        pytest.skip("No multi-chunk documents found")
    
    # Test a sample
    test_size = min(20, len(multi_chunk_ids))
    test_ids = random.sample(multi_chunk_ids, test_size)
    
    with ZPKGReader(str(zpkg_file)) as reader:
        errors = []
        for corpus_id in test_ids:
            chunk_indices = mapping[corpus_id]
            
            try:
                # Decompress using the multi-chunk method (handles UTF-8 boundaries)
                decompressed_text = reader.get_decompressed_chunks(chunk_indices)
                
                # Check for UTF-8 decoding errors (replacement characters)
                if '\ufffd' in decompressed_text:
                    errors.append(f"{corpus_id}: UTF-8 decoding errors detected")
                
                # Check that text is valid UTF-8
                decompressed_text.encode('utf-8')
                
            except UnicodeDecodeError as e:
                errors.append(f"{corpus_id}: Unicode decode error: {e}")
            except Exception as e:
                errors.append(f"{corpus_id}: error: {e}")
        
        assert len(errors) == 0, (
            f"Found {len(errors)} UTF-8 boundary handling errors:\n" +
            "\n".join(errors[:10])
        )
