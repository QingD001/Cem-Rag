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

from mteb import evaluate as mteb_evaluate, get_task
from mteb.cache import ResultCache

# Import Qwen embedding model from models module
from src.models.qwen_embedding import QwenEmbeddingModel
from src.models.e5_mistral_embedding import E5MistralEmbeddingModel
from src.utils import flatten_results_directory, log_gpu_memory, sort_top_level_keys

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
    
    os.makedirs(output_dir, exist_ok=True)
    
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
            # Determine model directory name for saving results
            model_dir_name = None
            if hasattr(model, 'mteb_model_meta') and model.mteb_model_meta:
                model_name = getattr(model.mteb_model_meta, 'name', None)
                if model_name:
                    model_dir_name = model_name.split('/')[1] if '/' in model_name else model_name
            
            if model_dir_name:
                model_dir = Path(output_dir) / model_dir_name
                model_dir.mkdir(parents=True, exist_ok=True)
                task_result_file = model_dir / f"{task_name}.json"
                
                # Try to load existing partial results if available
                # This allows us to preserve results from completed sub-tasks even if later ones fail
                existing_results = {}
                if task_result_file.exists():
                    try:
                        with open(task_result_file, 'r') as f:
                            existing_results = json.load(f)
                        logger.info(f"Found existing results file, will merge with new results")
                    except (json.JSONDecodeError, IOError) as e:
                        logger.warning(f"Could not read existing results file: {e}")
            
            # Use new mteb.evaluate API instead of deprecated MTEB class
            # Pass custom_cache to save results directly to our output directory
            task_results = mteb_evaluate(
                model=model,
                tasks=[task_obj],
                cache=custom_cache,
            )
            
            # Process results and save detailed results file
            for task_result in task_results:
                if task_result.task_name == task_name:
                    results_dict[task_name] = task_result.scores
                    logger.info(f"✓ {task_name} completed successfully")
                    
                    # Save detailed task result to JSON file
                    if model_dir_name:
                        # Try to get full result data from task_result object
                        # Check if task_result has a to_dict() method or similar
                        task_result_data = {}
                        if hasattr(task_result, 'to_dict'):
                            task_result_data = task_result.to_dict()
                        elif hasattr(task_result, '__dict__'):
                            # Try to serialize the object's attributes
                            task_result_data = {}
                            for key, value in task_result.__dict__.items():
                                try:
                                    # Try to serialize the value
                                    json.dumps(value)  # Test if serializable
                                    task_result_data[key] = value
                                except (TypeError, ValueError):
                                    # If not serializable, try to convert
                                    if hasattr(value, '__dict__'):
                                        task_result_data[key] = str(value)
                                    else:
                                        task_result_data[key] = str(value)
                        else:
                            # Fallback: construct from available attributes
                            task_result_data = {
                                "task_name": task_name,
                                "scores": task_result.scores.copy() if hasattr(task_result.scores, 'copy') else dict(task_result.scores),
                            }
                            # Add additional metadata if available
                            for attr in ['description', 'main_score', 'evaluation_time', 'mteb_version', 'dataset_revision']:
                                if hasattr(task_result, attr):
                                    task_result_data[attr] = getattr(task_result, attr)
                        
                        # Merge with existing results if available (preserve completed sub-tasks)
                        if existing_results and 'scores' in existing_results and 'scores' in task_result_data:
                            # Merge scores: new results take precedence, but keep existing sub-tasks that aren't in new results
                            existing_scores = existing_results.get('scores', {})
                            new_scores = task_result_data.get('scores', {})
                            merged_scores = {**existing_scores, **new_scores}  # New overwrites old
                            task_result_data['scores'] = merged_scores
                            logger.info(f"Merged with existing results, preserving completed sub-tasks")
                        
                        with open(task_result_file, 'w') as f:
                            json.dump(task_result_data, f, indent=2, default=str)
                        logger.info(f"Saved detailed result to: {task_result_file}")
                    
                    break
            
            # Flatten directory structure: move files from results/model/revision/ to model/
            # MTEB saves to output_dir/results/{model_name}/{revision}/{task_name}.json
            # We flatten this to output_dir/{model_name}/{task_name}.json
            if hasattr(model, 'mteb_model_meta') and model.mteb_model_meta and hasattr(model.mteb_model_meta, 'name'):
                model_name = model.mteb_model_meta.name
                flatten_results_directory(output_dir, model_name=model_name)
        
        except Exception as e:
            logger.error(f"✗ {task_name} failed: {e}")
            failed_tasks.append((task_name, str(e)))
            
            # Try to save partial results if available (e.g., from MTEB cache)
            # This is especially useful for tasks with multiple sub-tasks (like LEMBPasskeyRetrieval)
            # where some sub-tasks may complete before others fail
            if model_dir_name and task_result_file and task_result_file.exists():
                try:
                    # Check if MTEB saved partial results to cache
                    # MTEB may have saved results even if evaluation failed
                    with open(task_result_file, 'r') as f:
                        partial_results = json.load(f)
                    logger.info(f"Found partial results in {task_result_file}, they are preserved")
                except (json.JSONDecodeError, IOError):
                    pass
            
            continue
    
    # Report failed tasks
    if failed_tasks:
        logger.warning(f"\n{len(failed_tasks)} task(s) failed:")
        for task_name, error in failed_tasks:
            logger.warning(f"  - {task_name}: {error}")
    
    if not results_dict:
        return {}  # Return empty dict if all tasks failed
    
    output_dict = {}
    
    # Process each task result individually (synthetic vs real tasks)
    # scores structure: dict[SplitName, list[Scores]] where Scores is dict[str, Any]
    context_length_list = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    
    for key, value in results_dict.items():
        # Check if this specific task is synthetic (Needle/Passkey)
        is_synthetic_task = "Needle" in key or "Passkey" in key
        
        if is_synthetic_task:
            # Synthetic tasks: extract scores for different context lengths
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
        sorted_output = sort_top_level_keys(output_dict)
        print(json.dumps(sorted_output, indent=2))
        
        # Save summary to file
        # flatten_results_directory has already flattened to output_dir/{model_dir_name}/
        if hasattr(model, 'mteb_model_meta') and model.mteb_model_meta:
            model_name = getattr(model.mteb_model_meta, 'name', None)
            if model_name:
                # Extract model directory name (remove organization prefix if present)
                model_dir_name = model_name.split('/')[1] if '/' in model_name else model_name
                model_dir = Path(output_dir) / model_dir_name
                model_dir.mkdir(parents=True, exist_ok=True)
                
                summary_path = model_dir / "summary.json"
                # Merge with existing summary if it exists
                if summary_path.exists():
                    try:
                        with open(summary_path, 'r') as f:
                            existing_summary = json.load(f)
                        # Merge: existing tasks are preserved, new tasks are added/updated
                        existing_summary.update(output_dict)
                        output_dict = existing_summary
                    except (json.JSONDecodeError, IOError) as e:
                        logger.warning(f"Failed to read existing summary.json: {e}, overwriting")
                
                # Sort only top-level keys (task names), preserve nested structure
                sorted_output = sort_top_level_keys(output_dict)
                with open(summary_path, 'w') as f:
                    json.dump(sorted_output, f, indent=2)
                logger.info(f"Results saved to: {output_dir}/{model_dir_name}")
            else:
                logger.warning(f"Could not determine model directory name, results not saved to file")
        else:
            logger.warning(f"Model metadata not available, results not saved to file")
    else:
        logger.warning("No results to display (all tasks may have failed)")
    
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

