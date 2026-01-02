"""
zpkg Builder using zstd C API for real-time output monitoring

This implementation uses ctypes to call zstd C library directly,
enabling true real-time output size monitoring as described in the design.
"""

import ctypes
import ctypes.util
import struct
import json
import argparse
import logging
import pickle
from typing import BinaryIO, Dict, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ZPKGManifest:
    """Manifest information for a zpkg file"""
    zpkg_path: str
    num_chunks: int
    chunk_size: int
    dict_size: int
    total_compressed_size: int


# ZSTD constants
ZSTD_e_continue = 0
ZSTD_e_flush = 1
ZSTD_e_end = 2

ZSTD_c_compressionLevel = 100


class ZSTD_inBuffer(ctypes.Structure):
    _fields_ = [
        ("src", ctypes.POINTER(ctypes.c_char)),
        ("size", ctypes.c_size_t),
        ("pos", ctypes.c_size_t),
    ]


class ZSTD_outBuffer(ctypes.Structure):
    _fields_ = [
        ("dst", ctypes.POINTER(ctypes.c_char)),
        ("size", ctypes.c_size_t),
        ("pos", ctypes.c_size_t),
    ]


class ZPKGBuilder:
    """
    zpkg Builder using zstd C API for output-driven compression.
    
    This enables real-time output size monitoring, allowing us to
    control frame size based on compressed output rather than input.
    """
    
    MAGIC = b'ZPKG'
    HEADER_SIZE = 32
    VERSION = 0x01
    INPUT_CHUNK_SIZE = 1024  # Size of input chunks fed to compressor (1KB)
    
    def __init__(
        self,
        target_chunk_size: int = 4096,
        compression_level: int = 3,
        dict_size: int = 64 * 1024,  # Default: 64KB
        guard: int = 256,
    ):
        """
        Args:
            target_chunk_size: Target compressed chunk size
            compression_level: zstd compression level (1-22)
            dict_size: Size of shared dictionary
            guard: Guard value to avoid exceeding target
        """
        self.target_chunk_size = target_chunk_size
        self.compression_level = compression_level
        self.dict_size = dict_size
        self.guard = guard
        self.threshold = target_chunk_size - guard
        
        # Load zstd library
        lib_path = ctypes.util.find_library('zstd')
        if not lib_path:
            raise RuntimeError("libzstd not found. Please install zstd development library.")
        
        self.libzstd = ctypes.CDLL('libzstd.so.1')
        self._setup_functions()
    
    def build_zpkg(
        self,
        corpus_jsonl_path: str,
        output_path: str,
        training_multiplier: int = 100,
    ) -> ZPKGManifest:
        """
        Build zpkg using C API with real-time output monitoring.
        
        Args:
            corpus_jsonl_path: Path to corpus JSONL file
            output_path: Output zpkg file path
            training_multiplier: How many times dict_size to use for training (default: 100)
        """
        # Step 1: Read corpus and prepare mapping
        total_lines, dict_samples, corpus_docs_list, corpus_to_chunks = \
            self._read_corpus_and_prepare_mapping(corpus_jsonl_path)
        
        # Train dictionary
        logger.info("Training dictionary...")
        dictionary = self._train_dictionary(iter(dict_samples), training_multiplier=training_multiplier)
        dict_size = len(dictionary)
        logger.info(f"Dictionary trained: {dict_size} bytes ({dict_size//1024} KB)")
        
        # Step 2: Compress corpus with real-time mapping
        chunks = self._compress_corpus_with_mapping(
            corpus_jsonl_path,
            total_lines,
            dictionary,
            corpus_docs_list,
            corpus_to_chunks,
        )
        
        # Step 3: Write zpkg file
        total_compressed = self._write_zpkg_file(output_path, dictionary, dict_size, chunks)
        
        # Step 4: Save mapping
        self._save_mapping(output_path, corpus_to_chunks)
        
        return ZPKGManifest(
            zpkg_path=output_path,
            num_chunks=len(chunks),
            chunk_size=self.target_chunk_size,
            dict_size=dict_size,
            total_compressed_size=total_compressed,
        )
    
    # Private helper methods
    def _setup_functions(self):
        """Setup C function signatures"""
        # ZSTD_createCCtx
        self.libzstd.ZSTD_createCCtx.restype = ctypes.c_void_p
        self.libzstd.ZSTD_createCCtx.argtypes = []
        
        # ZSTD_freeCCtx
        self.libzstd.ZSTD_freeCCtx.argtypes = [ctypes.c_void_p]
        
        # ZSTD_compressStream2
        self.libzstd.ZSTD_compressStream2.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ZSTD_outBuffer),
            ctypes.POINTER(ZSTD_inBuffer),
            ctypes.c_int,
        ]
        self.libzstd.ZSTD_compressStream2.restype = ctypes.c_size_t
        
        # ZSTD_CCtx_setParameter
        self.libzstd.ZSTD_CCtx_setParameter.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.libzstd.ZSTD_CCtx_setParameter.restype = ctypes.c_size_t
        
        # ZSTD_CCtx_loadDictionary
        self.libzstd.ZSTD_CCtx_loadDictionary.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.libzstd.ZSTD_CCtx_loadDictionary.restype = ctypes.c_size_t
    
    def _train_dictionary(self, corpus_samples, training_multiplier: int = 100) -> bytes:
        """Train shared dictionary using zstd's training algorithm
        
        Args:
            corpus_samples: Iterator of training samples
            training_multiplier: How many times dict_size to use for training (default: 100)
        """
        import zstandard as zstd
        
        # Collect training samples
        samples = []
        total_size = 0
        target_size = self.dict_size * training_multiplier
        
        for sample in corpus_samples:
            samples.append(sample)
            total_size += len(sample)
            if total_size >= target_size:
                break
        
        if not samples:
            raise ValueError("No training samples provided")
        
        # Use zstd's actual dictionary training algorithm
        # This extracts representative patterns from the entire training set,
        # not just the first N bytes
        try:
            dictionary = zstd.train_dictionary(
                self.dict_size,
                samples,
                level=6  # Training level (higher = better quality but slower)
            )
            return dictionary.as_bytes()
        except AttributeError:
            # Fallback: if train_dictionary is not available, use simple approach
            # (but this should not happen with modern zstandard library)
            samples_bytes = bytearray()
            for sample in samples:
                samples_bytes.extend(sample)
                if len(samples_bytes) >= self.dict_size:
                    break
            
            if len(samples_bytes) > self.dict_size:
                return bytes(samples_bytes[:self.dict_size])
            else:
                return bytes(samples_bytes) + b'\x00' * (self.dict_size - len(samples_bytes))
    
    def _read_corpus_and_prepare_mapping(
        self,
        corpus_jsonl_path: str,
    ) -> Tuple[int, List[bytes], List[Tuple[str, bytes]], Dict[str, List[int]]]:
        """Read corpus, collect dictionary samples, and pre-load documents for mapping.
        
        Returns:
            Tuple of (total_lines, dict_samples, corpus_docs_list, corpus_to_chunks)
        """
        logger.info("Counting documents...")
        with open(corpus_jsonl_path, 'r') as f:
            total_lines = sum(1 for _ in f)
        
        logger.info(f"Reading {total_lines:,} documents...")
        dict_samples = []
        corpus_docs_list = []
        corpus_to_chunks: Dict[str, List[int]] = {}
        
        with open(corpus_jsonl_path, 'r') as f:
            for line in tqdm(f, total=total_lines, desc="Reading corpus", unit="docs"):
                doc = json.loads(line)
                corpus_id = doc.get('_id', '')
                text = doc.get('text', '') + '\n'
                text_bytes = text.encode('utf-8')
                
                # Collect samples for dictionary training
                dict_samples.append(text_bytes)
                
                # Pre-load documents for mapping
                corpus_docs_list.append((corpus_id, text_bytes))
                corpus_to_chunks[corpus_id] = []
        
        return total_lines, dict_samples, corpus_docs_list, corpus_to_chunks
    
    def _compress_corpus_with_mapping(
        self,
        corpus_jsonl_path: str,
        total_lines: int,
        dictionary: bytes,
        corpus_docs_list: List[Tuple[str, bytes]],
        corpus_to_chunks: Dict[str, List[int]],
    ) -> List[bytes]:
        """Compress corpus with real-time document-to-chunk mapping.
        
        Returns:
            List of compressed chunks
        """
        # Initialize compression state
        state = self._init_compression_state(dictionary)
        input_remainder = bytearray()
        
        # Main compression loop
        with open(corpus_jsonl_path, 'r') as f:
            for line in tqdm(f, total=total_lines, desc="Compressing", unit="docs"):
                doc = json.loads(line)
                text = doc.get('text', '') + '\n'
                text_bytes = text.encode()
                input_remainder.extend(text_bytes)
                
                # Process input in chunks
                while len(input_remainder) >= self.INPUT_CHUNK_SIZE:
                    input_chunk = bytes(input_remainder[:self.INPUT_CHUNK_SIZE])
                    input_remainder = input_remainder[self.INPUT_CHUNK_SIZE:]
                    
                    remaining_input = self._compress_input_chunk(state, input_chunk)
                    self._flush_compression_output(state)
                    self._check_and_end_chunk_if_needed(
                        state, state['chunks'], corpus_docs_list, corpus_to_chunks, dictionary, remaining_input
                    )
        
        # Process remaining input and finalize
        if len(input_remainder) > 0:
            self._process_remaining_input(state, input_remainder)
        
        self._finalize_last_chunk(state, state['chunks'], corpus_docs_list, corpus_to_chunks)
        
        # Cleanup compression context
        if state['cctx']:
            self.libzstd.ZSTD_freeCCtx(state['cctx'])
        
        return state['chunks']
    
    def _init_compression_state(self, dictionary: bytes):
        """Initialize compression state with buffers and context.
        
        Returns:
            A dictionary containing all compression state variables
        """
        output_buffer_size = self.target_chunk_size * 3
        output_buffer = (ctypes.c_char * output_buffer_size)()
        cctx = self._create_compression_context(dictionary)
        out_buf = ZSTD_outBuffer(
            ctypes.cast(output_buffer, ctypes.POINTER(ctypes.c_char)),
            output_buffer_size,
            0
        )
        
        return {
            'chunks': [],
            'chunk_idx': 0,
            'doc_ptr': 0,
            'chunk_decompressed_len': 0,
            'remaining_doc_len': 0,
            'output_buffer': output_buffer,
            'output_buffer_size': output_buffer_size,
            'cctx': cctx,
            'out_buf': out_buf,
        }
    
    def _create_compression_context(self, dictionary: bytes):
        """Create and initialize a new zstd compression context.
        
        Returns:
            Compression context (cctx)
        """
        cctx = self.libzstd.ZSTD_createCCtx()
        if not cctx:
            raise RuntimeError("Failed to create compression context")
        
        # Set compression level
        self.libzstd.ZSTD_CCtx_setParameter(
            cctx,
            ZSTD_c_compressionLevel,
            self.compression_level
        )
        
        # Load dictionary
        dict_buf = (ctypes.c_char * len(dictionary)).from_buffer_copy(dictionary)
        result = self.libzstd.ZSTD_CCtx_loadDictionary(
            cctx,
            ctypes.cast(dict_buf, ctypes.POINTER(ctypes.c_char)),
            len(dictionary)
        )
        if result != 0:
            raise RuntimeError(f"Failed to load dictionary: {result}")
        
        return cctx
    
    def _compress_input_chunk(self, state: dict, input_chunk: bytes) -> int:
        """Compress input chunk and track consumed bytes.
        
        Returns:
            Remaining input size (should be 0 in normal cases)
        """
        input_buf_data = (ctypes.c_char * len(input_chunk)).from_buffer_copy(input_chunk)
        in_buf = ZSTD_inBuffer(
            ctypes.cast(input_buf_data, ctypes.POINTER(ctypes.c_char)),
            len(input_chunk),
            0
        )
        
        while in_buf.pos < in_buf.size:
            consumed_before = in_buf.pos
            result = self.libzstd.ZSTD_compressStream2(
                state['cctx'],
                ctypes.byref(state['out_buf']),
                ctypes.byref(in_buf),
                ZSTD_e_continue
            )
            if result != 0:
                raise RuntimeError(f"Compression error: {result}")
            consumed = in_buf.pos - consumed_before
            state['chunk_decompressed_len'] += consumed
        
        return in_buf.size - in_buf.pos
    
    def _flush_compression_output(self, state: dict):
        """Flush compression output to force data to be written."""
        empty_in_buf = ZSTD_inBuffer(
            ctypes.cast((ctypes.c_char * 1)(), ctypes.POINTER(ctypes.c_char)),
            0,
            0
        )
        result = self.libzstd.ZSTD_compressStream2(
            state['cctx'],
            ctypes.byref(state['out_buf']),
            ctypes.byref(empty_in_buf),
            ZSTD_e_flush
        )
        if result != 0:
            raise RuntimeError(f"Flush error: {result}")
    
    def _check_and_end_chunk_if_needed(
        self,
        state: dict,
        chunks: List[bytes],
        corpus_docs_list: List[Tuple[str, bytes]],
        corpus_to_chunks: Dict[str, List[int]],
        dictionary: bytes,
        remaining_input: int = 0,
    ):
        """Check output size and end chunk if threshold reached."""
        # Check output size after flush
        if state['out_buf'].pos >= self.threshold:
            if remaining_input > 0:
                state['chunk_decompressed_len'] += remaining_input
            self._finalize_chunk_frame(state)
            self._end_chunk_and_start_new(
                state, chunks, corpus_docs_list, corpus_to_chunks, dictionary
            )
        
        # Check if output buffer is getting full
        elif state['out_buf'].pos >= state['output_buffer_size'] - self.INPUT_CHUNK_SIZE:
            empty_in_buf = ZSTD_inBuffer(
                ctypes.cast((ctypes.c_char * 1)(), ctypes.POINTER(ctypes.c_char)),
                0,
                0
            )
            while True:
                result = self.libzstd.ZSTD_compressStream2(
                    state['cctx'],
                    ctypes.byref(state['out_buf']),
                    ctypes.byref(empty_in_buf),
                    ZSTD_e_end
                )
                if result == 0:
                    break
            
            self._end_chunk_and_start_new(
                state, chunks, corpus_docs_list, corpus_to_chunks, dictionary
            )
    
    def _finalize_chunk_frame(self, state: dict):
        """Finalize current chunk frame using ZSTD_e_end."""
        end_empty_in_buf = ZSTD_inBuffer(
            ctypes.cast((ctypes.c_char * 1)(), ctypes.POINTER(ctypes.c_char)),
            0,
            0
        )
        while True:
            result = self.libzstd.ZSTD_compressStream2(
                state['cctx'],
                ctypes.byref(state['out_buf']),
                ctypes.byref(end_empty_in_buf),
                ZSTD_e_end
            )
            if result == 0:
                break
            if state['out_buf'].pos >= self.target_chunk_size:
                break
    
    def _end_chunk_and_start_new(
        self,
        state: dict,
        chunks: List[bytes],
        corpus_docs_list: List[Tuple[str, bytes]],
        corpus_to_chunks: Dict[str, List[int]],
        dictionary: bytes,
    ):
        """End current chunk, map documents, and start a new frame."""
        frame = bytes(state['output_buffer'][:state['out_buf'].pos])
        chunks.append(frame)
        self._map_documents_to_chunk(state, corpus_docs_list, corpus_to_chunks)
        state['out_buf'].pos = 0
        
        # Create new compression frame
        if state['cctx']:
            self.libzstd.ZSTD_freeCCtx(state['cctx'])
        state['cctx'] = self._create_compression_context(dictionary)
    
    def _map_documents_to_chunk(
        self,
        state: dict,
        corpus_docs_list: List[Tuple[str, bytes]],
        corpus_to_chunks: Dict[str, List[int]],
    ):
        """Map documents to current chunk based on decompressed length."""
        remaining_in_chunk = state['chunk_decompressed_len']
        
        # Handle remaining part of spanning document if any
        if state['remaining_doc_len'] > 0:
            if remaining_in_chunk >= state['remaining_doc_len']:
                if state['doc_ptr'] < len(corpus_docs_list):
                    corpus_id, _ = corpus_docs_list[state['doc_ptr']]
                    if corpus_id:
                        corpus_to_chunks[corpus_id].append(state['chunk_idx'])
                remaining_in_chunk -= state['remaining_doc_len']
                state['remaining_doc_len'] = 0
                state['doc_ptr'] += 1
            else:
                if state['doc_ptr'] < len(corpus_docs_list):
                    corpus_id, _ = corpus_docs_list[state['doc_ptr']]
                    if corpus_id:
                        corpus_to_chunks[corpus_id].append(state['chunk_idx'])
                state['remaining_doc_len'] -= remaining_in_chunk
                remaining_in_chunk = 0
        
        # Process new documents
        while state['doc_ptr'] < len(corpus_docs_list) and remaining_in_chunk > 0:
            corpus_id, doc_bytes = corpus_docs_list[state['doc_ptr']]
            doc_len = len(doc_bytes)
            
            if remaining_in_chunk >= doc_len:
                if corpus_id:
                    corpus_to_chunks[corpus_id].append(state['chunk_idx'])
                remaining_in_chunk -= doc_len
                state['doc_ptr'] += 1
            else:
                if corpus_id:
                    corpus_to_chunks[corpus_id].append(state['chunk_idx'])
                state['remaining_doc_len'] = doc_len - remaining_in_chunk
                break
        
        state['chunk_decompressed_len'] = 0
        state['chunk_idx'] += 1
    
    def _process_remaining_input(self, state: dict, input_remainder: bytearray):
        """Process remaining input data after main loop."""
        input_buf_data = (ctypes.c_char * len(input_remainder)).from_buffer_copy(bytes(input_remainder))
        in_buf = ZSTD_inBuffer(
            ctypes.cast(input_buf_data, ctypes.POINTER(ctypes.c_char)),
            len(input_remainder),
            0
        )
        
        while in_buf.pos < in_buf.size:
            consumed_before = in_buf.pos
            result = self.libzstd.ZSTD_compressStream2(
                state['cctx'],
                ctypes.byref(state['out_buf']),
                ctypes.byref(in_buf),
                ZSTD_e_continue
            )
            if result != 0:
                break
            consumed = in_buf.pos - consumed_before
            state['chunk_decompressed_len'] += consumed
    
    def _finalize_last_chunk(
        self,
        state: dict,
        chunks: List[bytes],
        corpus_docs_list: List[Tuple[str, bytes]],
        corpus_to_chunks: Dict[str, List[int]],
    ):
        """Finalize the last chunk frame."""
        empty_in_buf = ZSTD_inBuffer(
            ctypes.cast((ctypes.c_char * 1)(), ctypes.POINTER(ctypes.c_char)),
            0,
            0
        )
        while True:
            result = self.libzstd.ZSTD_compressStream2(
                state['cctx'],
                ctypes.byref(state['out_buf']),
                ctypes.byref(empty_in_buf),
                ZSTD_e_end
            )
            if result == 0:
                break
        
        final_frame = bytes(state['output_buffer'][:state['out_buf'].pos])
        if len(final_frame) > 0:
            chunks.append(final_frame)
            self._map_documents_to_chunk(state, corpus_docs_list, corpus_to_chunks)
    
    def _write_zpkg_file(
        self,
        output_path: str,
        dictionary: bytes,
        dict_size: int,
        chunks: List[bytes],
    ) -> int:
        """Write zpkg file with chunks and index table.
        
        Returns:
            Total compressed size
        """
        total_compressed = sum(len(c) for c in chunks)
        
        with open(output_path, 'wb') as f:
            # Write header (with placeholder for index_table_offset)
            self._write_header(f, dict_size, len(chunks), 0)
            
            # Write dictionary
            f.write(dictionary)
            
            # Write chunks and record offsets
            chunk_offsets = []
            chunks_start = f.tell()
            
            for chunk in chunks:
                offset = f.tell() - chunks_start
                chunk_offsets.append(offset)
                f.write(chunk)
            
            # Write index table (only offsets, uint32 each)
            index_start = f.tell()
            for offset in chunk_offsets:
                f.write(struct.pack('<I', offset))
            
            # Update header with index table offset
            current_pos = f.tell()
            f.seek(22)  # Position of index_table_offset in header (0x16)
            f.write(struct.pack('<Q', index_start))
            f.seek(current_pos)
        
        return total_compressed
    
    def _save_mapping(self, output_path: str, corpus_to_chunks: Dict[str, List[int]]):
        """Save corpus-to-chunks mapping to pickle file."""
        mapping_path = output_path + '.mapping.pkl'
        with open(mapping_path, 'wb') as f:
            pickle.dump(corpus_to_chunks, f)
        logger.info(f"Mapping saved: {mapping_path} ({len(corpus_to_chunks):,} corpus IDs)")
    
    def _write_header(self, f: BinaryIO, dict_size: int, num_chunks: int, index_table_offset: int = 0):
        """Write zpkg header
        
        Args:
            f: File handle
            dict_size: Dictionary size
            num_chunks: Number of chunks
            index_table_offset: Index table offset (will be filled in later if 0)
        """
        f.write(self.MAGIC)
        f.write(struct.pack('B', self.VERSION))
        f.write(struct.pack('B', 0))  # Reserved
        f.write(struct.pack('<I', self.target_chunk_size))
        f.write(struct.pack('<I', dict_size))
        f.write(struct.pack('<Q', num_chunks))
        f.write(struct.pack('<Q', index_table_offset))  # Index table offset (8 bytes)
        f.write(b'\x00' * 2)  # Reserved (2 bytes for future use)
    
def main():
    """Command-line interface for zpkg builder"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        datefmt='%H:%M:%S'
    )
    
    parser = argparse.ArgumentParser(
        description="Build zpkg file from corpus JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m zpkg.builder corpus.jsonl output.zpkg
  python -m zpkg.builder corpus.jsonl output.zpkg --dict-size 112 --compression-level 5
        """
    )
    
    parser.add_argument(
        'input',
        help='Input corpus JSONL file path'
    )
    parser.add_argument(
        'output',
        help='Output zpkg file path'
    )
    parser.add_argument(
        '--target-chunk-size',
        type=int,
        default=4096,
        help='Target compressed chunk size in bytes (default: 4096)'
    )
    parser.add_argument(
        '--compression-level',
        type=int,
        default=3,
        help='Zstd compression level 1-22 (default: 3)'
    )
    parser.add_argument(
        '--dict-size',
        type=int,
        default=64,
        help='Dictionary size in KB (default: 64)'
    )
    parser.add_argument(
        '--guard',
        type=int,
        default=256,
        help='Guard value to avoid exceeding target (default: 256)'
    )
    parser.add_argument(
        '--training-multiplier',
        type=int,
        default=100,
        help='How many times dict_size to use for training (default: 100)'
    )
    
    args = parser.parse_args()
    
    # Convert dict_size from KB to bytes
    dict_size_bytes = args.dict_size * 1024
    
    # Create builder
    builder = ZPKGBuilder(
        target_chunk_size=args.target_chunk_size,
        compression_level=args.compression_level,
        dict_size=dict_size_bytes,
        guard=args.guard
    )
    
    # Build zpkg
    manifest = builder.build_zpkg(
        corpus_jsonl_path=args.input,
        output_path=args.output,
        training_multiplier=args.training_multiplier
    )
    
    logger.info("Compression completed!")
    logger.info(f"  Output file: {args.output}")
    logger.info(f"  Total chunks: {manifest.num_chunks:,}")
    logger.info(f"  Total compressed size: {manifest.total_compressed_size:,} bytes ({manifest.total_compressed_size/1024/1024:.2f} MB)")


if __name__ == '__main__':
    main()
