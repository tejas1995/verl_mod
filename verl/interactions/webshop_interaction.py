# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import sys
import yaml
import argparse
from typing import Any, Optional, Dict, List, Tuple
from uuid import uuid4

# Add the parent directory to sys.path to import from core
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from core.envs import get_env
from core.evaluators import get_evaluators
from core.user_simulator import get_user_simulator
from core.scenarios import WebShopScenario

from .base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class WebShopInteraction(BaseInteraction):
    """Multi-turn webshop interaction for VERL training.
    
    This class extends VERL's BaseInteraction to handle multi-turn conversations
    in the webshop environment, integrating with existing environment, user simulator,
    and evaluator components.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}
        
        # Initialize components based on config
        self._init_components(config)
        
    def _init_components(self, config: dict):
        """Initialize environment, user simulator, and evaluators."""
        # Create args object from config
        args = argparse.Namespace()
        args.scenarios_file = config.get("scenarios_file")  # Optional - scenarios can come from dataset
        args.env_type = config.get("env_type", "webshop")
        args.evaluator_model_name = config.get("evaluator_model_name", "claude-3-7-sonnet-latest")
        
        # Initialize environment
        self.env = get_env(args.env_type, args)
        # Scenarios will be loaded from dataset via interaction_kwargs
        self.scenarios_file = None
        
        # Initialize user simulator - require user_config path
        user_config_path = config.get("user_config")
        assert user_config_path is not None, "user_config must be provided as a path to a YAML config file"
        assert os.path.exists(user_config_path), f"user_config file not found: {user_config_path}"
        
        with open(user_config_path, "r") as f:
            user_config = yaml.safe_load(f)
        self.user_simulator = get_user_simulator(user_config, args)
        
        logger.info(f"Initialized WebShopInteraction with env_type={args.env_type}")
    
    def _ensure_scenarios_loaded(self, scenarios_file: str = None):
        """Ensure scenarios are loaded from the specified file."""
        if scenarios_file and scenarios_file != self.scenarios_file:
            # Load scenarios from the specified file (from dataset)
            if hasattr(self.env, 'load_scenarios'):
                self.env.load_scenarios(scenarios_file)
                self.scenarios_file = scenarios_file
        elif not hasattr(self.env, 'scenarios') or not self.env.scenarios:
            # No scenarios loaded yet, require scenarios_file from dataset
            if scenarios_file and hasattr(self.env, 'load_scenarios'):
                self.env.load_scenarios(scenarios_file)
                self.scenarios_file = scenarios_file
            else:
                raise ValueError("No scenarios loaded and no scenarios_file provided. Scenarios must be provided via interaction_kwargs.")

    async def start_interaction(
        self, 
        instance_id: Optional[str] = None, 
        scenario_data: Optional[dict] = None,
        scenario_index: Optional[int] = None,
        **kwargs
    ) -> str:
        """Initialize a new webshop interaction session.
        
        Args:
            instance_id: Unique identifier for this interaction instance
            scenario_data: Scenario data dictionary (if provided, scenario_index is ignored)
            scenario_index: Index of scenario to use (if scenario_data not provided)
            **kwargs: Additional arguments
            
        Returns:
            The instance_id for tracking this interaction
        """
        if instance_id is None:
            instance_id = str(uuid4())
            
        # Determine scenario index to use
        if scenario_index is not None:
            final_scenario_index = scenario_index
        elif kwargs.get("scenario_index") is not None:
            # Get scenario index from interaction_kwargs (from dataset)
            final_scenario_index = kwargs["scenario_index"]
        else:
            # Use first scenario as default
            final_scenario_index = 0
            
        # Initialize interaction state
        self._instance_dict[instance_id] = {
            "scenario_index": final_scenario_index,
            "trajectory": [],
            "num_environment_steps": 0,
            "num_agent_utterances": 0,
            "num_errors": 0,
            "done": False,
            "info": {},
            "turn_rewards": [],  # Track rewards for each turn
            "episode_reward": 0.0,
            "max_env_steps": kwargs.get("max_env_steps", 50),
            "max_agent_utterances": kwargs.get("max_agent_utterances", 20),
        }
        
        # Ensure scenarios are loaded (from dataset file if provided)
        scenarios_file = kwargs.get("scenarios_file")
        self._ensure_scenarios_loaded(scenarios_file)
        # Initialize environment with the determined scenario index
        self.env.initialize_environment(final_scenario_index)
        
        # Reset user simulator with scenario
        scenario_obj = self.env.get_current_scenario()
        if hasattr(scenario_obj, 'scenario_data'):
            scenario_data_for_user = scenario_obj.scenario_data
        else:
            scenario_data_for_user = scenario_obj
        self.user_simulator.reset(scenario_data_for_user)
        
        # Add initial user message to trajectory
        initial_user_message = self.user_simulator.get_user_utterance()
        self._instance_dict[instance_id]["trajectory"].append({
            "role": "user", 
            "content": f"USER INITIAL INSTRUCTION: {initial_user_message}"
        })
        
        logger.info(f"Started interaction {instance_id} with scenario {scenario_data_for_user.get('scenario_id', 'unknown')}")
        return instance_id

    async def generate_response(
        self, 
        instance_id: str, 
        messages: list[dict[str, Any]], 
        **kwargs
    ) -> Tuple[bool, str, float, dict[str, Any]]:
        """Extract agent action from messages and provide environment feedback.
        
        This method does NOT generate the agent response - the agent (LLM) does that.
        Instead, it extracts the agent's action from the conversation messages,
        executes it in the WebShop environment, and returns environment feedback.
        
        Args:
            instance_id: Unique identifier for this interaction instance
            messages: List of conversation messages (agent response is already included)
            **kwargs: Additional arguments
            
        Returns:
            Tuple containing:
            - should_terminate_sequence (bool): True if interaction should end
            - response_content (str): Environment feedback (becomes next user message)
            - current_turn_score (float): Reward for this turn
            - additional_data (dict): Additional metadata
        """
        if instance_id not in self._instance_dict:
            raise ValueError(f"Instance {instance_id} not found")
            
        instance = self._instance_dict[instance_id]
        
        # Check if we should terminate due to limits
        if (instance["num_environment_steps"] >= instance["max_env_steps"] or 
            instance["num_agent_utterances"] >= instance["max_agent_utterances"] or
            instance["done"]):
            return True, "Interaction terminated due to limits or completion", 0.0, instance["info"]
        
        try:
            # Extract agent action from messages
            agent_output = self._extract_agent_output(messages)
            if not agent_output:
                return False, "No valid agent output found", -0.1, {}
            
            # Validate and extract action
            action, action_tag = self.env.extract_preproc_and_validate_action(agent_output)
            instance["trajectory"].append({
                "role": "assistant", 
                "content": agent_output, 
                "action_tag": action_tag
            })
            
            # Calculate turn-level reward (simple, fast)
            turn_reward = await self._calculate_turn_reward(instance, action, action_tag)
            instance["turn_rewards"].append(turn_reward)
            
            # Handle different action types
            if action.startswith("respond["):
                # Agent is responding to user
                logger.info(f"Utterance #{instance['num_agent_utterances']}: {action} (action_type: {action_tag})")
                agent_message = action.split("respond[")[-1].strip("]").strip()
                user_response = self.user_simulator.get_user_utterance(agent_message)
                instance["trajectory"].append({
                    "role": "user", 
                    "content": f"USER RESPONSE: {user_response}"
                })
                instance["num_agent_utterances"] += 1
                response_content = f"User responded: {user_response}"
                
            else:
                # Agent is taking environment action
                logger.info(f"Action #{instance['num_environment_steps']}: {action} (action_type: {action_tag})")
                obs, reward, done, info = self.env.step(action)
                instance["trajectory"].append({"role": "user", "content": obs})
                instance["num_environment_steps"] += 1
                instance["done"] = done
                instance["info"] = info
                response_content = obs
                
                # Add environment reward to turn reward
                if reward is not None:
                    turn_reward += reward
                    instance["turn_rewards"][-1] = turn_reward
            
            # Check if episode is done
            should_terminate = (instance["done"] or 
                              instance["num_environment_steps"] >= instance["max_env_steps"] or
                              instance["num_agent_utterances"] >= instance["max_agent_utterances"])
            
            additional_data = {
                "action": action,
                "action_tag": action_tag,
                "num_environment_steps": instance["num_environment_steps"],
                "num_agent_utterances": instance["num_agent_utterances"],
                "done": instance["done"],
                "info": instance["info"]
            }
            
            return should_terminate, response_content, turn_reward, additional_data
            
        except Exception as e:
            instance["num_errors"] += 1
            logger.error(f"Error in interaction {instance_id}: {e}")
            
            # If too many errors, terminate
            if instance["num_errors"] >= 10:
                return True, f"Interaction terminated due to errors: {str(e)}", -1.0, {}
            
            return False, f"Error occurred: {str(e)}", -0.5, {"error": str(e)}

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        """Calculate episode-level reward using the WebShop reward function.
        
        Simplified version that directly calls webshop.compute_score().
        
        Args:
            instance_id: Unique identifier for this interaction instance
            **kwargs: Additional arguments
            
        Returns:
            Episode reward score
        """
        if instance_id not in self._instance_dict:
            return 0.0
            
        instance = self._instance_dict[instance_id]
        
        # If episode is not done, return current cumulative reward
        if not instance["done"]:
            return sum(instance["turn_rewards"])
        
        # Use the WebShop reward function (same as training)
        if instance["info"].get("purchased_item") is not None:
            from verl.utils.reward_score.webshop import compute_score
            
            # Get scenario data from environment
            scenario_obj = self.env.get_current_scenario()
            if hasattr(scenario_obj, 'scenario_data'):
                scenario_data = scenario_obj.scenario_data
            else:
                scenario_data = scenario_obj
                
            # Prepare extra_info for reward function
            extra_info = {
                "purchased_product": instance["info"]["purchased_item"],
                "trajectory": instance["trajectory"],
            }
            
            # Call the reward function with weights as kwargs
            reward_result = compute_score(
                "webshop", 
                None, 
                json.dumps(scenario_data), 
                extra_info,
                task_weight=kwargs["task_weight"],
                satisfaction_weight=kwargs["satisfaction_weight"],
                evaluator_model_name=kwargs["evaluator_model_name"],
                secrets_file=kwargs["secrets_file"]
            )
            return reward_result["score"]
        else:
            return 0.0

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """Clean up interaction resources.
        
        Args:
            instance_id: Unique identifier for this interaction instance
            **kwargs: Additional arguments
        """
        if instance_id in self._instance_dict:
            # Log final statistics
            instance = self._instance_dict[instance_id]
            logger.info(f"Finalized interaction {instance_id}: "
                       f"steps={instance['num_environment_steps']}, "
                       f"utterances={instance['num_agent_utterances']}, "
                       f"errors={instance['num_errors']}, "
                       f"reward={instance.get('episode_reward', 0.0)}")
            
            # Clean up instance data
            del self._instance_dict[instance_id]

    def _extract_agent_output(self, messages: list[dict]) -> str:
        """Extract agent output from VERL messages format.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Agent output string
        """
        # Find the last assistant message
        for message in reversed(messages):
            if message.get("role") == "assistant":
                return message.get("content", "")
        return ""

    async def _calculate_turn_reward(self, instance: dict, action: str, action_tag: str) -> float:
        """Calculate simple reward for a single turn.
        
        Args:
            instance: Interaction instance data
            action: Action taken by agent
            action_tag: Tag associated with action
            
        Returns:
            Simple turn-level reward (0.0 to 0.1)
        """
        # Simple rule-based turn rewards
        # if action.startswith("search["):
        #     return 0.05  # Small reward for searching
        # elif action.startswith("click["):
        #     return 0.1   # Slightly higher reward for clicking
        # elif action.startswith("respond["):
        #     return 0.02  # Small reward for responding
        # else:
        #     return 0.0   # No reward for other actions
        # TODO: Implement this
        return 0.0


    def get_interaction_stats(self, instance_id: str) -> dict:
        """Get statistics for a specific interaction instance.
        
        Args:
            instance_id: Unique identifier for this interaction instance
            
        Returns:
            Dictionary containing interaction statistics
        """
        if instance_id not in self._instance_dict:
            return {}
            
        instance = self._instance_dict[instance_id]
        scenario_obj = self.env.get_current_scenario()
        scenario_id = scenario_obj.scenario_id if hasattr(scenario_obj, 'scenario_id') else "unknown"
        
        return {
            "scenario_id": scenario_id,
            "num_environment_steps": instance["num_environment_steps"],
            "num_agent_utterances": instance["num_agent_utterances"],
            "num_errors": instance["num_errors"],
            "done": instance["done"],
            "episode_reward": instance.get("episode_reward", 0.0),
            "turn_rewards": instance["turn_rewards"],
            "trajectory_length": len(instance["trajectory"])
        }
    
