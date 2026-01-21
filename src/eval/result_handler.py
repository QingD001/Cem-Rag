"""
Result Handler for Evaluation Results

Handles saving, loading, and processing of evaluation results.
Separated from evaluation logic to follow single responsibility principle.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class ResultHandler:
    """Handles saving and processing evaluation results"""
    
    def __init__(self, output_dir: str, model_name: Optional[str] = None):
        """
        Initialize result handler.
        
        Args:
            output_dir: Base output directory for results
            model_name: Model name (e.g., "Qwen/Qwen3-Embedding-0.6B") for determining subdirectory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.model_dir_name = None
        self.model_dir = None
        
        if model_name:
            # Extract model directory name (remove organization prefix if present)
            self.model_dir_name = model_name.split('/')[1] if '/' in model_name else model_name
            self.model_dir = self.output_dir / self.model_dir_name
            self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def get_task_result_path(self, task_name: str) -> Path:
        """Get path for task result file"""
        if not self.model_dir:
            raise ValueError("Model directory not initialized")
        return self.model_dir / f"{task_name}.json"
    
    def load_existing_results(self, task_name: str) -> Dict:
        """
        Load existing results for a task if available.
        
        Args:
            task_name: Name of the task
            
        Returns:
            Dictionary with existing results, or empty dict if not found
        """
        task_result_file = self.get_task_result_path(task_name)
        if task_result_file.exists():
            try:
                with open(task_result_file, 'r') as f:
                    existing_results = json.load(f)
                logger.info(f"Found existing results file for {task_name}, will merge with new results")
                return existing_results
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not read existing results file {task_result_file}: {e}")
        return {}
    
    def save_task_result(self, task_name: str, task_result: Any, existing_results: Optional[Dict] = None):
        """
        Save detailed task result to JSON file.
        
        Args:
            task_name: Name of the task
            task_result: Task result object from MTEB
            existing_results: Optional existing results to merge with
        """
        if not self.model_dir:
            logger.warning(f"Model directory not initialized, skipping save for {task_name}")
            return
        
        task_result_file = self.get_task_result_path(task_name)
        
        # Extract result data from task_result object
        task_result_data = self._extract_result_data(task_result)
        
        # Merge with existing results if available
        if existing_results and 'scores' in existing_results and 'scores' in task_result_data:
            existing_scores = existing_results.get('scores', {})
            new_scores = task_result_data.get('scores', {})
            merged_scores = {**existing_scores, **new_scores}  # New overwrites old
            task_result_data['scores'] = merged_scores
            logger.info(f"Merged with existing results for {task_name}, preserving completed sub-tasks")
        
        # Save to file
        with open(task_result_file, 'w') as f:
            json.dump(task_result_data, f, indent=2, default=str)
        logger.info(f"Saved detailed result to: {task_result_file}")
    
    def _extract_result_data(self, task_result: Any) -> Dict:
        """Extract serializable data from task_result object"""
        if hasattr(task_result, 'to_dict'):
            return task_result.to_dict()
        elif hasattr(task_result, '__dict__'):
            task_result_data = {}
            for key, value in task_result.__dict__.items():
                try:
                    # Test if serializable
                    json.dumps(value)
                    task_result_data[key] = value
                except (TypeError, ValueError):
                    # Convert non-serializable values to string
                    if hasattr(value, '__dict__'):
                        task_result_data[key] = str(value)
                    else:
                        task_result_data[key] = str(value)
            return task_result_data
        else:
            # Fallback: construct from available attributes
            task_result_data = {
                "task_name": getattr(task_result, 'task_name', 'unknown'),
                "scores": getattr(task_result, 'scores', {}).copy() if hasattr(getattr(task_result, 'scores', {}), 'copy') else dict(getattr(task_result, 'scores', {})),
            }
            # Add additional metadata if available
            for attr in ['description', 'main_score', 'evaluation_time', 'mteb_version', 'dataset_revision']:
                if hasattr(task_result, attr):
                    task_result_data[attr] = getattr(task_result, attr)
            return task_result_data
    
    def flatten_results(self):
        """Flatten MTEB results directory structure"""
        if self.model_name:
            self._flatten_results_directory(str(self.output_dir), model_name=self.model_name)
    
    def _flatten_results_directory(self, output_dir: str, model_name: str):
        """
        Flatten MTEB results directory structure.
        
        When using custom ResultCache with cache_path=output_dir, MTEB saves results to:
        output_dir/results/{model_name}/{revision}/{task_name}.json
        
        This function flattens the structure to:
        output_dir/{model_name}/{task_name}.json
        
        Structure before: output_dir/results/Qwen__Qwen3-Embedding-0.6B/no_revision_available/file.json
        Structure after:  output_dir/Qwen3-Embedding-0.6B/file.json
        
        Args:
            output_dir: Output directory containing MTEB results
            model_name: Model name in MTEB format (e.g., "Qwen/Qwen3-Embedding-0.6B").
        """
        output_path = Path(output_dir)
        if not output_path.exists():
            return
        
        # Convert model name to MTEB directory format (e.g., "Qwen/Qwen3-Embedding-0.6B" -> "Qwen__Qwen3-Embedding-0.6B")
        mteb_dir_name = model_name.replace('/', '__')
        
        # With custom ResultCache, MTEB saves to: output_dir/results/{model_name}/{revision}/
        results_path = output_path / "results"
        model_dir = results_path / mteb_dir_name if results_path.exists() else None
        
        if not model_dir or not model_dir.exists():
            return
        
        # Process the model directory
        original_name = model_dir.name
        
        # First, flatten revision subdirectories in the current directory
        # MTEB typically creates revision subdirectories like 'no_revision_available'
        for revision_dir in list(model_dir.iterdir()):
            if revision_dir.is_dir():
                self._flatten_revision_subdirectory(revision_dir, model_dir)
        
        # Remove organization prefix from directory name
        # (e.g., "Qwen__Qwen3-Embedding-0.6B" -> "Qwen3-Embedding-0.6B")
        parts = original_name.split('__')
        if len(parts) < 2:
            return
        
        new_name = '__'.join(parts[1:])
        if not new_name or new_name == original_name:
            return
        
        # Move from output_dir/results/{model_name} to output_dir/{new_name}
        new_model_dir = output_path / new_name
        
        if new_model_dir.exists():
            # Target exists: flatten it first, then merge
            for target_revision_dir in list(new_model_dir.iterdir()):
                if target_revision_dir.is_dir():
                    self._flatten_revision_subdirectory(target_revision_dir, new_model_dir)
            self._merge_directories(model_dir, new_model_dir)
            logger.debug(f"Merged directory: {original_name} -> {new_name}")
        else:
            # Move from results/ subdirectory to output_dir root
            shutil.move(str(model_dir), str(new_model_dir))
            logger.debug(f"Moved directory from results/: {original_name} -> {new_name}")
    
    def _move_item_with_overwrite(self, item: Path, target: Path):
        """Move item to target, overwriting if target exists."""
        if target.exists():
            if target.is_file() and item.is_file():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)
        
        if item.is_file():
            item.rename(target)
        elif item.is_dir():
            shutil.move(str(item), str(target))
    
    def _flatten_revision_subdirectory(self, revision_dir: Path, target_dir: Path):
        """Flatten a revision subdirectory by moving all items to target directory."""
        moved_any = False
        for item in revision_dir.iterdir():
            target = target_dir / item.name
            self._move_item_with_overwrite(item, target)
            moved_any = True
        
        if moved_any:
            try:
                revision_dir.rmdir()
                logger.debug(f"Flattened directory: removed {revision_dir} and moved files to {target_dir}")
            except OSError:
                pass
    
    def _merge_directories(self, source_dir: Path, target_dir: Path):
        """Merge contents from source_dir to target_dir, overwriting conflicts."""
        for item in list(source_dir.iterdir()):
            target_item = target_dir / item.name
            self._move_item_with_overwrite(item, target_item)
        
        try:
            source_dir.rmdir()
            logger.debug(f"Merged and removed directory: {source_dir}")
        except OSError:
            logger.warning(f"Could not remove directory {source_dir} after merging (not empty)")
    
    def process_and_save_summary(self, results_dict: Dict):
        """
        Process results dictionary and save summary.
        
        Args:
            results_dict: Dictionary mapping task names to their scores
        """
        if not results_dict:
            logger.warning("No results to process")
            return
        
        output_dict = self._process_results(results_dict)
        
        if output_dict:
            # Print summary to console
            logger.info("\n" + "=" * 80)
            logger.info("Evaluation Summary")
            logger.info("=" * 80)
            sorted_output = self._sort_top_level_keys(output_dict)
            print(json.dumps(sorted_output, indent=2))
            
            # Save summary to file
            self._save_summary(output_dict)
        else:
            logger.warning("No results to display (all tasks may have failed)")
    
    def _process_results(self, results_dict: Dict) -> Dict:
        """
        Process raw results dictionary into summary format.
        
        Args:
            results_dict: Raw results from MTEB
            
        Returns:
            Processed summary dictionary
        """
        output_dict = {}
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
        
        return output_dict
    
    def _save_summary(self, output_dict: Dict):
        """Save summary dictionary to summary.json file"""
        if not self.model_dir:
            logger.warning("Model directory not initialized, summary not saved")
            return
        
        summary_path = self.model_dir / "summary.json"
        
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
        sorted_output = self._sort_top_level_keys(output_dict)
        with open(summary_path, 'w') as f:
            json.dump(sorted_output, f, indent=2)
        logger.info(f"Results saved to: {self.output_dir}/{self.model_dir_name}")
    
    def _sort_top_level_keys(self, data: Dict) -> Dict:
        """Sort only the top-level keys of a dictionary, preserving nested structure."""
        return {key: data[key] for key in sorted(data.keys())}
