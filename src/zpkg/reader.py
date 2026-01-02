"""
zpkg Reader: Reads compressed chunks from zpkg files

Design:
- Supports random access to compressed chunks
- Can decompress individual chunks on demand
- Variable-length chunks (requires index table or sequential scan)
"""

import zstandard as zstd
import struct
import os
import pickle
from typing import Optional, BinaryIO, List, Dict


class ZPKGReader:
    """
    Reads zpkg files and provides access to compressed chunks.
    
    For variable-length chunks, we have two options:
    1. Build index table on first access (lazy)
    2. Store index table in file (requires file format change)
    
    Currently using option 1: lazy index building.
    """
    
    MAGIC = b'ZPKG'
    HEADER_SIZE = 32
    
    def __init__(self, zpkg_path: str, mapping_path: Optional[str] = None):
        """
        Args:
            zpkg_path: Path to zpkg file
            mapping_path: Optional path to mapping file (.mapping.pkl). 
                         If None, will try to auto-detect by appending .mapping.pkl to zpkg_path
        """
        self.zpkg_path = zpkg_path
        self.file = None
        self._header = None
        self._dictionary = None
        self._chunk_offsets = None  # Lazy-loaded index
        self._decompressor = None
        self._mapping = None  # corpus_id -> List[chunk_indices]
        self._mapping_path = mapping_path
    
    def _ensure_open(self):
        """Ensure file is open"""
        if self.file is None:
            self.file = open(self.zpkg_path, 'rb')
        if self._header is None:
            self._read_header()
    
    def _read_header(self):
        """Read and parse zpkg header"""
        self.file.seek(0)
        
        # Read header
        magic = self.file.read(4)
        if magic != self.MAGIC:
            raise ValueError(f"Invalid zpkg file: magic mismatch")
        
        version = struct.unpack('B', self.file.read(1))[0]
        _reserved1 = struct.unpack('B', self.file.read(1))[0]
        target_chunk_size = struct.unpack('<I', self.file.read(4))[0]
        dict_size = struct.unpack('<I', self.file.read(4))[0]
        num_chunks = struct.unpack('<Q', self.file.read(8))[0]
        index_table_offset = struct.unpack('<Q', self.file.read(8))[0]  # Index table offset
        _reserved2 = self.file.read(2)  # Reserved (2 bytes)
        
        self._header = {
            'version': version,
            'target_chunk_size': target_chunk_size,
            'dict_size': dict_size,
            'num_chunks': num_chunks,
            'index_table_offset': index_table_offset,
        }
        self._chunk_sizes = None
        
        # Read dictionary
        self._dictionary = self.file.read(dict_size)
        if len(self._dictionary) != dict_size:
            raise ValueError(f"Failed to read dictionary: expected {dict_size}, got {len(self._dictionary)}")
        
        # Dictionary start position
        self._dict_start = self.HEADER_SIZE
        self._chunks_start = self.HEADER_SIZE + dict_size
    
    def _build_chunk_index(self):
        """Load chunk index from file (lazy loading)"""
        if self._chunk_offsets is not None:
            return
        
        self._ensure_open()
        
        # Read index table offset from header
        index_table_offset = self._header['index_table_offset']
        num_chunks = self._header['num_chunks']
        
        # Read index table (only offsets, uint32 each)
        self.file.seek(index_table_offset)
        
        chunk_offsets = []
        for _ in range(num_chunks):
            offset = struct.unpack('<I', self.file.read(4))[0]  # uint32 instead of uint64
            chunk_offsets.append(offset)
        
        # Calculate chunk sizes from offsets
        # Note: offsets are relative to chunks_start
        chunks_start = self.HEADER_SIZE + self._header['dict_size']
        chunk_sizes = []
        for i in range(len(chunk_offsets)):
            if i < len(chunk_offsets) - 1:
                # Size = next offset - current offset
                size = chunk_offsets[i + 1] - chunk_offsets[i]
            else:
                # Last chunk: size = (index_table_offset - chunks_start) - last_offset
                size = (index_table_offset - chunks_start) - chunk_offsets[i]
            chunk_sizes.append(size)
        
        self._chunk_offsets = chunk_offsets
        self._chunk_sizes = chunk_sizes
    
    def get_compressed_chunk(self, chunk_index: int) -> bytes:
        """
        Get compressed bytes for a chunk.
        
        Args:
            chunk_index: Index of chunk (0-based)
            
        Returns:
            Compressed bytes for the chunk
        """
        self._ensure_open()
        self._build_chunk_index()
        
        if chunk_index >= len(self._chunk_offsets):
            raise IndexError(f"Chunk index {chunk_index} out of range (max: {len(self._chunk_offsets) - 1})")
        
        chunks_start = self.HEADER_SIZE + self._header['dict_size']
        chunk_offset = chunks_start + self._chunk_offsets[chunk_index]
        chunk_size = self._chunk_sizes[chunk_index]
        
        self.file.seek(chunk_offset)
        return self.file.read(chunk_size)
    
    def get_decompressed_chunk_bytes(self, chunk_index: int) -> bytes:
        """
        Get decompressed bytes for a chunk (before UTF-8 decoding).
        
        Args:
            chunk_index: Index of chunk (0-based)
            
        Returns:
            Decompressed bytes
        """
        compressed = self.get_compressed_chunk(chunk_index)
        
        # Create decompressor if needed
        if self._decompressor is None:
            self._ensure_open()
            decompression_dict = zstd.ZstdCompressionDict(self._dictionary)
            self._decompressor = zstd.ZstdDecompressor(dict_data=decompression_dict)
        
        # Decompress using stream reader to handle frames without content size
        import io
        output = io.BytesIO()
        with self._decompressor.stream_reader(compressed) as reader:
            while True:
                chunk = reader.read(8192)
                if not chunk:
                    break
                output.write(chunk)
        
        return output.getvalue()
    
    def get_decompressed_chunk(self, chunk_index: int) -> str:
        """
        Get decompressed text for a chunk.
        
        Args:
            chunk_index: Index of chunk (0-based)
            
        Returns:
            Decompressed text
        """
        decompressed_bytes = self.get_decompressed_chunk_bytes(chunk_index)
        return decompressed_bytes.decode('utf-8', errors='replace')
    
    def get_decompressed_chunks(self, chunk_indices: List[int]) -> str:
        """
        Get decompressed text for multiple chunks, properly handling UTF-8 boundaries.
        
        This method concatenates the bytes first, then decodes to avoid UTF-8
        multi-byte character corruption at chunk boundaries.
        
        Args:
            chunk_indices: List of chunk indices (0-based)
            
        Returns:
            Decompressed text from all chunks combined
        """
        all_bytes = b""
        for chunk_index in chunk_indices:
            chunk_bytes = self.get_decompressed_chunk_bytes(chunk_index)
            all_bytes += chunk_bytes
        
        return all_bytes.decode('utf-8', errors='replace')
    
    @property
    def num_chunks(self) -> int:
        """Number of chunks in the zpkg file"""
        self._ensure_open()
        return self._header['num_chunks']
    
    @property
    def dict_size(self) -> int:
        """Size of the dictionary in bytes"""
        self._ensure_open()
        return self._header['dict_size']
    
    @property
    def target_chunk_size(self) -> int:
        """Target chunk size (compressed)"""
        self._ensure_open()
        return self._header['target_chunk_size']
    
    @property
    def version(self) -> int:
        """File format version"""
        self._ensure_open()
        return self._header['version']
    
    def _load_mapping(self):
        """Load mapping file (lazy loading)"""
        if self._mapping is not None:
            return
        
        # Auto-detect mapping path if not provided
        if self._mapping_path is None:
            self._mapping_path = self.zpkg_path + '.mapping.pkl'
        
        if not os.path.exists(self._mapping_path):
            self._mapping = {}  # Empty mapping if file doesn't exist
            return
        
        try:
            with open(self._mapping_path, 'rb') as f:
                self._mapping = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load mapping file {self._mapping_path}: {e}")
    
    def get_chunks_by_corpus_id(self, corpus_id: str) -> List[int]:
        """
        Get chunk indices for a given corpus_id.
        
        Args:
            corpus_id: Corpus document ID
            
        Returns:
            List of chunk indices (0-based) containing this corpus
            
        Raises:
            KeyError: If corpus_id not found in mapping
            ValueError: If mapping file not found or cannot be loaded
        """
        self._load_mapping()
        
        if corpus_id not in self._mapping:
            raise KeyError(f"Corpus ID '{corpus_id}' not found in mapping")
        
        return self._mapping[corpus_id]
    
    def get_document_by_corpus_id(self, corpus_id: str) -> str:
        """
        Get decompressed document text by corpus_id.
        
        Args:
            corpus_id: Corpus document ID
            
        Returns:
            Decompressed document text
            
        Raises:
            KeyError: If corpus_id not found in mapping
            ValueError: If mapping file not found or cannot be loaded
        """
        chunk_indices = self.get_chunks_by_corpus_id(corpus_id)
        
        if len(chunk_indices) == 1:
            return self.get_decompressed_chunk(chunk_indices[0])
        else:
            return self.get_decompressed_chunks(chunk_indices)
    
    def has_mapping(self) -> bool:
        """
        Check if mapping file is available.
        
        Returns:
            True if mapping file exists and can be loaded
        """
        try:
            self._load_mapping()
            return len(self._mapping) > 0
        except:
            return False
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the zpkg file.
        
        Returns:
            Dictionary with statistics including:
            - num_chunks: Number of chunks
            - dict_size: Dictionary size in bytes
            - target_chunk_size: Target chunk size
            - total_compressed_size: Total compressed size of all chunks
            - avg_chunk_size: Average compressed chunk size
            - file_size: Total file size
            - compression_ratio: Estimated compression ratio (if available)
        """
        self._ensure_open()
        self._build_chunk_index()
        
        total_compressed = sum(self._chunk_sizes)
        avg_chunk_size = total_compressed / len(self._chunk_sizes) if self._chunk_sizes else 0
        
        file_size = os.path.getsize(self.zpkg_path)
        
        stats = {
            'num_chunks': self.num_chunks,
            'dict_size': self.dict_size,
            'target_chunk_size': self.target_chunk_size,
            'total_compressed_size': total_compressed,
            'avg_chunk_size': avg_chunk_size,
            'file_size': file_size,
            'version': self.version,
        }
        
        # Add mapping statistics if available
        if self.has_mapping():
            self._load_mapping()
            stats['mapped_documents'] = len(self._mapping)
            multi_chunk_docs = sum(1 for chunks in self._mapping.values() if len(chunks) > 1)
            stats['multi_chunk_documents'] = multi_chunk_docs
            stats['single_chunk_documents'] = len(self._mapping) - multi_chunk_docs
        
        return stats
    
    def close(self):
        """Close the zpkg file"""
        if self.file is not None:
            self.file.close()
            self.file = None
    
    def __enter__(self):
        self._ensure_open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_compressed_chunk(zpkg_path: str, chunk_index: int) -> bytes:
    """Convenience function to get compressed chunk"""
    with ZPKGReader(zpkg_path) as reader:
        return reader.get_compressed_chunk(chunk_index)


def get_decompressed_chunk(zpkg_path: str, chunk_index: int) -> str:
    """Convenience function to get decompressed chunk"""
    with ZPKGReader(zpkg_path) as reader:
        return reader.get_decompressed_chunk(chunk_index)

