"""
WebShop reward function for VERL integration.

This module provides reward calculation for WebShop multi-turn agentic environments,
integrating existing task and satisfaction evaluators.
"""

import json
import logging
import argparse
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def compute_score(
    data_source: str, 
    solution_str: str, 
    ground_truth: str, 
    extra_info: Optional[Dict[str, Any]] = None, 
    task_weight: float = None, 
    satisfaction_weight: float = None, 
    evaluator_model_name: str = None,
    secrets_file: str = None
) -> Dict[str, Any]:
    """
    Compute reward score for WebShop interactions.
    
    This function integrates the existing WebShop evaluators (task and satisfaction)
    to provide comprehensive rewards for VERL training.
    
    Args:
        data_source: Dataset source identifier (should be "webshop" for WebShop scenarios)
        solution_str: The complete interaction trajectory as a string
        ground_truth: The scenario/goal data as a string (JSON format)
        extra_info: Additional information including:
            - purchased_product: Dict containing purchased product details
            - trajectory: List of conversation messages
            - scenario: Dict containing scenario data
            
    Returns:
        Dictionary containing:
            - score: Overall reward score (0-1)
            - task_score: Task completion score (0-1) 
            - satisfaction_score: User satisfaction score (0-1)
            - task_details: Detailed task evaluation results
            - satisfaction_details: Detailed satisfaction evaluation results
    """
    # Assert that all required keys are in extra_info
    required_keys = ["purchased_product", "trajectory"]
    for key in required_keys:
        if key not in extra_info:
            logger.warning(f"Missing required key: {key}")
            return {"score": 0.0, "error": f"Missing required key: {key}"}
    
    if data_source != "webshop":
        logger.warning(f"Unexpected data_source '{data_source}' for WebShop reward function")
        return {"score": 0.0, "error": f"Unsupported data_source: {data_source}"}
    
    if not extra_info:
        logger.warning("No extra_info provided to WebShop reward function")
        return {"score": 0.0, "error": "Missing extra_info"}
    
    try:
        # Parse ground truth (scenario data)
        scenario = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
        
        # Extract components from extra_info
        purchased_product = extra_info["purchased_product"]
        trajectory = extra_info["trajectory"]
        
        if not purchased_product:
            logger.warning("No purchased_product in extra_info")
            return {"score": 0.0, "error": "No purchased product to evaluate"}
        
        # Create evaluators using the provided model name and secrets file
        from core.evaluators.webshop.task_evaluator import WebShopTaskEvaluator
        from core.evaluators.webshop.satisfaction_evaluator import WebShopUserSatisfactionEvaluator
        import argparse
        
        # Create args object for evaluators
        args = argparse.Namespace()
        args.evaluator_model_name = evaluator_model_name
        args.secrets_file = secrets_file
        
        task_evaluator = WebShopTaskEvaluator(model_name=args.evaluator_model_name, args=args)
        satisfaction_evaluator = WebShopUserSatisfactionEvaluator(model_name=args.evaluator_model_name, args=args)
        
        # Calculate task reward
        task_results = task_evaluator.evaluate(purchased_product, scenario)
        task_score = task_results["total_reward"]
        
        # Calculate satisfaction reward  
        satisfaction_results = satisfaction_evaluator.evaluate(purchased_product, scenario, trajectory)
        satisfaction_score = _extract_satisfaction_score(satisfaction_results)
        
        # Combine scores (weighted average)
        # Use weights passed as function arguments
        
        # Validate weights
        assert task_weight >= 0, f"task_weight must be non-negative, got {task_weight}"
        assert satisfaction_weight >= 0, f"satisfaction_weight must be non-negative, got {satisfaction_weight}"
        
        total_weight = task_weight + satisfaction_weight
        assert total_weight > 0, f"Sum of weights must be positive, got task_weight={task_weight}, satisfaction_weight={satisfaction_weight}"
        
        # Normalize weights to ensure they sum to 1.0
        task_weight = task_weight / total_weight
        satisfaction_weight = satisfaction_weight / total_weight
        
        overall_score = (task_weight * task_score) + (satisfaction_weight * satisfaction_score)
        
        return {
            "score": overall_score,
            "task_score": task_score,
            "satisfaction_score": satisfaction_score,
            "task_eval_details": task_results,
            "satisfaction_eval_details": satisfaction_results,
            "weights": {
                "task_weight": task_weight,
                "satisfaction_weight": satisfaction_weight
            }
        }
        
    except Exception as e:
        logger.error(f"Error computing WebShop reward: {e}")
        return {"score": 0.0, "error": str(e)}


def _extract_satisfaction_score(satisfaction_results: Dict[str, Any]) -> float:
    """
    Extract overall satisfaction score from evaluation results.
    
    Args:
        satisfaction_results: User satisfaction evaluation results
        
    Returns:
        Overall satisfaction score (0-1)
    """
    # if not satisfaction_results:
    #     return 0.0
        
    # # Extract scores from different criteria
    # scores = []
    # for key, value in satisfaction_results.items():
    #     if isinstance(value, dict) and "score" in value:
    #         score = value["score"]
    #         # Normalize scores to 0-1 range
    #         if isinstance(score, (int, float)):
    #             if score > 5:  # Likert scale 1-5
    #                 score = score / 5.0
    #             scores.append(score)
    
    # return sum(scores) / len(scores) if scores else 0.0
    # TODO: Implement this
    return 0.0