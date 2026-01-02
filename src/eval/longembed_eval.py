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

import os
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

try:
    # Use MTEB class (mteb.evaluate doesn't support custom model classes well)
    # We'll use MTEB class with task objects instead of task names
    from mteb import MTEB, get_task
except ImportError:
    print("Error: mteb is not installed. Please install it with:")
    print("  pip install mteb")
    print("\nFor LongEmbed support, you may need to install from source:")
    print("  pip install git+https://github.com/embeddings-benchmark/mteb.git")
    sys.exit(1)

# Import Qwen embedding model from models module
from src.models.qwen_embedding import QwenEmbeddingModel

# Get logger (configuration should be done in main() or entry point)
logger = logging.getLogger(__name__)


def _flatten_results_directory(output_dir: str):
    """
    Flatten MTEB results directory structure by moving files from model/revision/ to model/.
    Also removes the organization prefix from directory names (e.g., Qwen__Qwen3-Embedding-0.6B -> Qwen3-Embedding-0.6B).
    
    Structure before: results/Qwen__Qwen3-Embedding-0.6B/no_revision_available/file.json
    Structure after:  results/Qwen3-Embedding-0.6B/file.json
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return
    
    # Find all model directories
    for model_dir in output_path.iterdir():
        if not model_dir.is_dir() or model_dir.name.startswith('.'):
            continue
        
        # Remove organization prefix (e.g., "Qwen__Qwen3-Embedding-0.6B" -> "Qwen3-Embedding-0.6B")
        # MTEB converts "Qwen/Qwen3-Embedding-0.6B" to "Qwen__Qwen3-Embedding-0.6B"
        # We want to remove the "Qwen__" prefix
        original_name = model_dir.name
        if '__' in original_name:
            # Split by '__' and take the last part (model name without org prefix)
            parts = original_name.split('__')
            if len(parts) >= 2:
                # Check if the first part looks like an organization name (short, capitalized)
                # and the rest looks like a model name
                new_name = '__'.join(parts[1:])  # Join all parts after the first
                if new_name and new_name != original_name:
                    new_model_dir = model_dir.parent / new_name
                    # Rename the directory
                    model_dir.rename(new_model_dir)
                    model_dir = new_model_dir
                    logger.debug(f"Renamed model directory: {original_name} -> {new_name}")
        
        # Look for revision subdirectories (like 'no_revision_available' or any other)
        for revision_dir in model_dir.iterdir():
            if not revision_dir.is_dir():
                continue
            
            # Move all files from revision subdirectory to model directory
            moved_any = False
            for item in revision_dir.iterdir():
                target = model_dir / item.name
                if target.exists():
                    # If target exists, check if it's the same file (same size and mtime)
                    # If different, log a warning but still overwrite (later revision wins)
                    if target.is_file() and item.is_file():
                        target_stat = target.stat()
                        item_stat = item.stat()
                        if target_stat.st_size != item_stat.st_size or target_stat.st_mtime != item_stat.st_mtime:
                            logger.warning(f"Overwriting existing file {target.name} from {revision_dir.name} "
                                         f"(size: {target_stat.st_size} -> {item_stat.st_size} bytes)")
                        target.unlink()
                    elif target.is_dir():
                        import shutil
                        logger.warning(f"Overwriting existing directory {target.name} from {revision_dir.name}")
                        shutil.rmtree(target)
                
                item.rename(target)
                moved_any = True
            
            # Remove empty revision directory
            if moved_any:
                try:
                    revision_dir.rmdir()
                    logger.debug(f"Flattened directory: removed {revision_dir} and moved files to {model_dir}")
                except OSError:
                    # Directory not empty, skip
                    pass


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
    
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    for task_obj in task_objects:
        task_name = task_obj.metadata.name
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Evaluating {task_name}...")
        logger.info(f"{'=' * 80}")
        
        try:
            # Create a single-task evaluation
            evaluation = MTEB(tasks=[task_obj])
            if not hasattr(evaluation, 'tasks') or not evaluation.tasks:
                evaluation.tasks = [task_obj]
            
            # Run evaluation for this task
            task_results = evaluation.run(
                model,
                output_folder=output_dir,
                overwrite_results=True,
                batch_size=batch_size,
                verbosity=0,
            )
            
            # Process results
            for task_result in task_results:
                if task_result.task_name == task_name:
                    results_dict[task_name] = task_result.scores
                    logger.info(f"✓ {task_name} completed successfully")
                    break
            
            # Flatten directory structure: move files from model/revision/ to model/
            # This removes the unnecessary revision subdirectory
            _flatten_results_directory(output_dir)
        
        except Exception as e:
            logger.error(f"✗ {task_name} failed: {e}")
            failed_tasks.append((task_name, str(e)))
            continue
    
    # Report failed tasks
    if failed_tasks:
        logger.warning(f"\n{len(failed_tasks)} task(s) failed:")
        for task_name, error in failed_tasks:
            logger.warning(f"  - {task_name}: {error}")
    
    if not results_dict:
        logger.error("\n" + "=" * 80)
        logger.error("All tasks failed! Cannot proceed with evaluation.")
        logger.error("=" * 80)
        logger.error("\nPossible reasons:")
        logger.error("  1. Dataset not downloaded (check HuggingFace cache)")
        logger.error("  2. Network issues preventing dataset download")
        logger.error("  3. Dataset configuration mismatch")
        logger.error("\nTo download datasets, you may need to:")
        logger.error("  - Set HF_TOKEN environment variable for authentication")
        logger.error("  - Run evaluation with network access enabled")
        logger.error("  - Check if datasets are available on HuggingFace Hub")
        return {}  # Return empty dict instead of raising exception
    
    output_dict = {}
    
    # Check if tasks are synthetic (Needle/Passkey) or real tasks
    is_synthetic = any("Needle" in task or "Passkey" in task for task in tasks)
    
    if is_synthetic:
        # Synthetic tasks: extract scores for different context lengths
        # scores structure: dict[SplitName, list[Scores]] where Scores is dict[str, Any]
        context_length_list = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
        for key, value in results_dict.items():
            needle_passkey_score_list = []
            for ctx_len in context_length_list:
                test_key = f"test_{ctx_len}"
                if test_key in value:
                    # value[test_key] is a list of score dicts, get the first one
                    score_list = value[test_key]
                    if score_list and isinstance(score_list, list) and len(score_list) > 0:
                        score_dict = score_list[0]  # Get first score dict
                        if "ndcg_at_1" in score_dict:
                            needle_passkey_score_list.append([ctx_len, score_dict["ndcg_at_1"]])
            if needle_passkey_score_list:
                avg_score = sum([x[1] for x in needle_passkey_score_list]) / len(needle_passkey_score_list)
                needle_passkey_score_list.append(["avg", avg_score])
                output_dict[key] = {item[0]: item[1] for item in needle_passkey_score_list}
    else:
        # Real tasks: extract ndcg@1 and ndcg@10
        # scores structure: dict[SplitName, list[Scores]] where Scores is dict[str, Any]
        for key, value in results_dict.items():
            split = "test" if "test" in value else "validation"
            if split in value:
                # value[split] is a list of score dicts, get the first one
                score_list = value[split]
                if score_list and isinstance(score_list, list) and len(score_list) > 0:
                    score_dict = score_list[0]  # Get first score dict
                    output_dict[key] = {
                        "ndcg@1": score_dict.get("ndcg_at_1", 0.0),
                        "ndcg@10": score_dict.get("ndcg_at_10", 0.0)
                    }
    
    # Print summary to console (much shorter than full MTEB results)
    # Full results are already saved by MTEB to output_dir (results/)
    if output_dict:
        logger.info("\n" + "=" * 80)
        logger.info("Evaluation Summary")
        logger.info("=" * 80)
        print(json.dumps(output_dict, indent=2))
        
        # Show output directory (simplified - just show the directory, not all files)
        logger.info("\n" + "=" * 80)
        logger.info("Results saved to:")
        logger.info("=" * 80)
        
        output_path = Path(output_dir)
        if output_path.exists():
            # Find model-specific directories
            model_dirs = [d for d in output_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
            if model_dirs:
                for model_dir in sorted(model_dirs):
                    # Count task result files (exclude model_meta.json)
                    task_files = [f for f in model_dir.glob("*.json") 
                                 if f.name != "model_meta.json" and f.is_file()]
                    if task_files:
                        logger.info(f"  {model_dir.name}/ ({len(task_files)} task result files)")
                    else:
                        logger.info(f"  {model_dir.name}/ (no task results found)")
            else:
                logger.info(f"  {output_dir}")
        else:
            logger.warning(f"  Output directory does not exist: {output_dir}")
    else:
        logger.warning("No results to display (all tasks may have failed)")
        logger.info(f"Check MTEB output directory: {output_dir}")
    
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
        help="Model version for Qwen ('0.6b', '4b', '8b') or path to model directory"
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
    model = QwenEmbeddingModel(version=args.model, device=args.device, project_root=project_root)
    
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

