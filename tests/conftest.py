"""
Pytest configuration and shared fixtures for zpkg tests
"""
import pytest
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from src.zpkg.reader import ZPKGReader


def find_latest_files(pattern_base: str, output_dir: str = "output") -> Tuple[Optional[Path], Optional[Path]]:
    """
    Find latest mapping and zpkg files matching pattern.
    
    Args:
        pattern_base: Base pattern to match (e.g., "nq", "trec-covid")
        output_dir: Directory to search in
        
    Returns:
        Tuple of (mapping_file_path, zpkg_file_path) or (None, None) if not found
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return None, None
    
    # Find all matching mapping files
    mapping_files = list(output_path.glob(f"*{pattern_base}*.mapping.pkl"))
    if not mapping_files:
        return None, None
    
    # Use the latest one
    mapping_file = max(mapping_files, key=lambda p: p.stat().st_mtime)
    
    # Find corresponding zpkg file
    zpkg_path = mapping_file.with_suffix('').with_suffix('.zpkg')
    if not zpkg_path.exists():
        # Try alternative search
        zpkg_files = list(output_path.glob(f"*{pattern_base}*.zpkg"))
        if zpkg_files:
            zpkg_path = max(zpkg_files, key=lambda p: p.stat().st_mtime)
        else:
            return None, None
    
    return mapping_file, zpkg_path


def load_corpus_map(corpus_jsonl_path: str) -> Dict[str, bytes]:
    """
    Load original corpus and build corpus_id -> content mapping.
    
    Args:
        corpus_jsonl_path: Path to corpus JSONL file
        
    Returns:
        Dictionary mapping corpus_id to content bytes
    """
    corpus_map = {}
    with open(corpus_jsonl_path, 'r') as f:
        for line in f:
            doc = json.loads(line)
            corpus_id = doc.get('_id', '')
            text = doc.get('text', '') + '\n'
            if corpus_id:
                corpus_map[corpus_id] = text.encode('utf-8')
    return corpus_map


@pytest.fixture
def sample_mapping_file(tmp_path, request):
    """
    Fixture to provide a mapping file path.
    Can be overridden by test markers or parameters.
    """
    # If test has a marker specifying pattern, use it
    pattern = getattr(request.node, 'pattern', None)
    if pattern:
        mapping_file, _ = find_latest_files(pattern)
        if mapping_file:
            return mapping_file
    
    # Default: return None, tests should handle this
    return None


@pytest.fixture
def sample_zpkg_file(tmp_path, request):
    """
    Fixture to provide a zpkg file path.
    Can be overridden by test markers or parameters.
    """
    pattern = getattr(request.node, 'pattern', None)
    if pattern:
        _, zpkg_file = find_latest_files(pattern)
        if zpkg_file:
            return zpkg_file
    
    return None


@pytest.fixture
def sample_corpus_path(request):
    """
    Fixture to provide corpus JSONL path.
    Can be overridden by test markers.
    """
    corpus_path = getattr(request.node, 'corpus_path', None)
    return corpus_path


@pytest.fixture
def zpkg_reader(sample_zpkg_file, sample_mapping_file):
    """
    Fixture providing a ZPKGReader instance.
    """
    if sample_zpkg_file is None:
        pytest.skip("No zpkg file available")
    
    reader = ZPKGReader(str(sample_zpkg_file), 
                        str(sample_mapping_file) if sample_mapping_file else None)
    yield reader
    reader.close()


@pytest.fixture
def corpus_mapping(sample_corpus_path):
    """
    Fixture providing corpus_id -> content mapping.
    """
    if sample_corpus_path is None:
        pytest.skip("No corpus path available")
    
    return load_corpus_map(sample_corpus_path)


@pytest.fixture
def mapping_dict(sample_mapping_file):
    """
    Fixture providing the mapping dictionary (corpus_id -> List[chunk_indices]).
    """
    if sample_mapping_file is None:
        pytest.skip("No mapping file available")
    
    with open(sample_mapping_file, 'rb') as f:
        return pickle.load(f)
