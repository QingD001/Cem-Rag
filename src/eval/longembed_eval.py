#!/usr/bin/env python3
"""
LongEmbed Benchmark Evaluation Script

This script evaluates embedding models on the LongEmbed benchmark,
which includes tasks for long-context retrieval as described in:
"LongEmbed: Extending Embedding Models for Long Context Retrieval"
https://arxiv.org/pdf/2404.12096

The benchmark includes:
- 2 synthetic tasks: Passkey, Needle
- 4 real-world tasks: NarrativeQA, QMSum, SummScreenFD, 2WikiMultihopQA
"""

import sys
from pathlib import Path

# Add project root to path for imports (if running as script)
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import json
import logging
from typing import Dict, List, Optional

from mteb import evaluate as mteb_evaluate, get_task
from mteb.cache import ResultCache

# Import Qwen embedding model from models module
from src.models.qwen_embedding import QwenEmbeddingModel
from src.models.e5_mistral_embedding import E5MistralEmbeddingModel
from src.utils import log_gpu_memory
from src.eval.result_handler import ResultHandler

# Get logger (configuration should be done in main() or entry point)
logger = logging.getLogger(__name__)


def evaluate_longembed(
    model,
    output_dir: Optional[str] = None,
    tasks: Optional[List[str]] = None,
    batch_size: int = 32,
) -> Dict:
    """
    Evaluate any embedding model on LongEmbed benchmark.
    
    The model must implement:
    - encode(texts: List[str], **kwargs) -> np.ndarray
    - encode_queries(queries: List[str], **kwargs) -> np.ndarray (optional, defaults to encode)
    - encode_corpus(corpus: List[str], **kwargs) -> np.ndarray (optional, defaults to encode)
    
    Args:
        model: Model object with encode methods (e.g., QwenEmbeddingModel)
        output_dir: Directory to save results. If None, uses "results/{model_name}"
        tasks: List of task names to evaluate. If None, evaluates all LongEmbed tasks
        batch_size: Batch size for encoding
        
    Returns:
        Dictionary with evaluation results
    """
    if output_dir is None:
        # Use simple results directory - MTEB will create subdirectories automatically
        output_dir = "results"
    
    # Initialize result handler
    model_name = None
    if hasattr(model, 'mteb_model_meta') and model.mteb_model_meta:
        model_name = getattr(model.mteb_model_meta, 'name', None)
    result_handler = ResultHandler(output_dir, model_name=model_name)
    
    # Create a custom ResultCache with our output directory as the cache path
    # This allows mteb.evaluate to save results directly to our desired output directory
    # instead of the default ~/.cache/mteb/results/
    custom_cache = ResultCache(cache_path=output_dir)
    
    # Default LongEmbed tasks if not specified
    # According to official LongEmbed repo: https://github.com/dwzhu-pku/LongEmbed
    # Task names should be: LEMBSummScreenFDRetrieval, LEMBQMSumRetrieval, LEMBWikimQARetrieval, 
    # LEMBNarrativeQARetrieval, LEMBNeedleRetrieval, LEMBPasskeyRetrieval
    if tasks is None:
        # Use official task names from LongEmbed repository
        tasks = [
            "LEMBSummScreenFDRetrieval",
            "LEMBQMSumRetrieval",
            "LEMBWikimQARetrieval",
            "LEMBNarrativeQARetrieval",
            "LEMBNeedleRetrieval",
            "LEMBPasskeyRetrieval",
        ]
        logger.info("Using official LongEmbed task names from https://github.com/dwzhu-pku/LongEmbed")
    
    logger.info(f"Evaluating on {len(tasks)} tasks: {tasks}")
    
    # Convert task names to task objects (MTEB requires task objects, not strings)
    task_objects = []
    for task_name in tasks:
        try:
            task_obj = get_task(task_name)
            task_objects.append(task_obj)
        except Exception as e:
            logger.warning(f"Failed to load task {task_name}: {e}")
            raise
    
    # Run tasks individually to handle failures gracefully
    # This allows other tasks to continue even if one fails
    results_dict = {}
    failed_tasks = []
    
    for idx, task_obj in enumerate(task_objects, 1):
        task_name = task_obj.metadata.name
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Evaluating {task_name} ({idx}/{len(task_objects)})...")
        logger.info(f"{'=' * 80}")
        
        # Filter out test_32768 split for synthetic tasks to avoid OOM
        # This applies to LEMBPasskeyRetrieval and LEMBNeedleRetrieval
        if "Passkey" in task_name or "Needle" in task_name:
            # Try to filter splits through different possible attributes
            filtered = False
            if hasattr(task_obj, 'evaluation_splits') and task_obj.evaluation_splits:
                original_splits = list(task_obj.evaluation_splits) if not isinstance(task_obj.evaluation_splits, list) else task_obj.evaluation_splits.copy()
                # Remove test_32768 split
                filtered_splits = [s for s in original_splits if s != "test_32768"]
                if len(filtered_splits) < len(original_splits):
                    task_obj.evaluation_splits = filtered_splits
                    filtered = True
            elif hasattr(task_obj, 'metadata') and hasattr(task_obj.metadata, 'eval_splits'):
                original_splits = list(task_obj.metadata.eval_splits) if not isinstance(task_obj.metadata.eval_splits, list) else task_obj.metadata.eval_splits.copy()
                filtered_splits = [s for s in original_splits if s != "test_32768"]
                if len(filtered_splits) < len(original_splits):
                    task_obj.metadata.eval_splits = filtered_splits
                    filtered = True
            
            if filtered:
                logger.info(f"Skipping test_32768 split for {task_name} to avoid OOM")
                logger.info(f"Will evaluate splits: {filtered_splits if 'filtered_splits' in locals() else 'N/A'}")
        
        try:
            # Load existing results if available
            existing_results = result_handler.load_existing_results(task_name)
            
            # Use new mteb.evaluate API instead of deprecated MTEB class
            # Pass custom_cache to save results directly to our output directory
            task_results = mteb_evaluate(
                model=model,
                tasks=[task_obj],
                cache=custom_cache,
            )
            
            # Process results
            for task_result in task_results:
                if task_result.task_name == task_name:
                    results_dict[task_name] = task_result.scores
                    logger.info(f"✓ {task_name} completed successfully")
                    
                    # Save detailed task result using result handler
                    result_handler.save_task_result(task_name, task_result, existing_results=existing_results)
                    break
            
            # Flatten directory structure after each task
            result_handler.flatten_results()
        
        except Exception as e:
            logger.error(f"✗ {task_name} failed: {e}")
            failed_tasks.append((task_name, str(e)))
            
            # Try to save partial results if available (e.g., from MTEB cache)
            # This is especially useful for tasks with multiple sub-tasks (like LEMBPasskeyRetrieval)
            # where some sub-tasks may complete before others fail
            try:
                task_result_file = result_handler.get_task_result_path(task_name)
                if task_result_file.exists():
                    with open(task_result_file, 'r') as f:
                        partial_results = json.load(f)
                    logger.info(f"Found partial results in {task_result_file}, they are preserved")
            except (json.JSONDecodeError, IOError, ValueError):
                pass
            
            continue
    
    # Report failed tasks
    if failed_tasks:
        logger.warning(f"\n{len(failed_tasks)} task(s) failed:")
        for task_name, error in failed_tasks:
            logger.warning(f"  - {task_name}: {error}")
    
    if not results_dict:
        return {}  # Return empty dict if all tasks failed
    
    # Process and save summary using result handler
    result_handler.process_and_save_summary(results_dict)
    
    # Log GPU memory after evaluation
    if hasattr(model, 'device') and model.device.startswith("cuda"):
        log_gpu_memory(model.device, "GPU memory after evaluation")
    
    return results_dict


def main():
    """Main entry point"""
    import argparse
    
    # Configure logging only once at application entry point
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    parser = argparse.ArgumentParser(description="Evaluate embedding model on LongEmbed benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="0.6b",
        help="Model version: '0.6b', '4b', '8b' (Qwen models) or 'e5-mistral' (E5-Mistral-7B-Instruct)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: results/{model_version})"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Specific tasks to evaluate (default: all LongEmbed tasks)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run model on ('cuda', 'cpu', 'cuda:0', 'cuda:1', etc.). "
             "If 'cuda' is specified, will auto-select GPU with most free memory."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("LongEmbed Benchmark Evaluation")
    logger.info("=" * 80)
    logger.info(f"Model version: {args.model}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Initialize model
    project_root = Path(__file__).parent.parent.parent
    
    # Determine which model to use based on model argument
    if args.model.lower() == "e5-mistral":
        model = E5MistralEmbeddingModel(device=args.device, project_root=project_root)
    else:
        model = QwenEmbeddingModel(version=args.model, device=args.device, project_root=project_root)
    
    # Log GPU memory before evaluation
    if args.device.startswith("cuda") or args.device == "cuda":
        log_gpu_memory(model.device if hasattr(model, 'device') else args.device, "GPU memory before evaluation")
    
    # Run evaluation
    # evaluate_longembed already prints the summary, no need to print again
    results = evaluate_longembed(
        model=model,
        output_dir=args.output_dir,
        tasks=args.tasks,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

