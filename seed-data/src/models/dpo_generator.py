"""DPO dataset generation pipeline."""

import ast
import time
import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
from omegaconf import DictConfig

from src.models.llm_client import LLMClient
from src.utils.prompts import get_answer_system_message, get_judge_system_message
from src.utils.text_processing import extract_answer_num, format_problem_prompt

logger = logging.getLogger(__name__)


class DPODatasetGenerator:
    """Generate DPO (Direct Preference Optimization) dataset from problems."""
    
    def __init__(self, cfg: DictConfig, llm_client: LLMClient):
        """
        Initialize generator.
        
        Args:
            cfg: Hydra configuration
            llm_client: LLM client instance
        """
        self.cfg = cfg
        self.client = llm_client
        self.answer_system_msg = get_answer_system_message()
        self.judge_system_msg = get_judge_system_message()
        
    def generate_answer(
        self,
        user_content: str,
        model_config: DictConfig
    ) -> str:
        """
        Generate answer using specified model configuration.
        
        Args:
            user_content: Problem prompt
            model_config: Model configuration
            
        Returns:
            Model response
        """
        messages = [
            {"role": "system", "content": self.answer_system_msg},
            {"role": "user", "content": user_content}
        ]
        
        extra_body = dict(model_config.get("extra_body", {}))
        
        return self.client.call(
            messages=messages,
            model_name=model_config.name,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            top_p=model_config.top_p,
            extra_body=extra_body if extra_body else None
        )
        
    def judge_answers(self, user_content: str, answer_a: str, answer_b: str) -> str:
        """
        Judge which answer has better reasoning.
        
        Args:
            user_content: Problem prompt
            answer_a: First answer
            answer_b: Second answer
            
        Returns:
            Judgment result ('A', 'B', or 'TIE')
        """
        judge_prompt = f"""### Question
{user_content}

### Answer A
{answer_a}

### Answer B
{answer_b}"""
        
        messages = [
            {"role": "system", "content": self.judge_system_msg},
            {"role": "user", "content": judge_prompt}
        ]
        
        judge_config = self.cfg.model.judge_model
        return self.client.call(
            messages=messages,
            model_name=judge_config.name,
            temperature=judge_config.temperature,
            max_tokens=judge_config.max_tokens,
            top_p=judge_config.top_p
        )
        
    def select_chosen_rejected(
        self,
        answer_a: str,
        answer_b: str,
        ans_num_a: int,
        ans_num_b: int,
        gold: int,
        user_content: str
    ) -> Tuple[Optional[str], Optional[str], str]:
        """
        Select chosen and rejected answers based on correctness and judging.
        
        Args:
            answer_a: First model's answer
            answer_b: Second model's answer
            ans_num_a: Extracted answer number from A
            ans_num_b: Extracted answer number from B
            gold: Gold standard answer
            user_content: Problem prompt (for judging)
            
        Returns:
            Tuple of (chosen, rejected, strategy_note)
        """
        # Case 1: Only A is correct
        if ans_num_a == gold and ans_num_b != gold:
            return answer_a, answer_b, "A_Correct"
            
        # Case 2: Only B is correct
        elif ans_num_b == gold and ans_num_a != gold:
            return answer_b, answer_a, "B_Correct"
            
        # Case 3: Both correct - use judge
        elif ans_num_a == gold and ans_num_b == gold:
            if not self.cfg.strategy.use_judge:
                return answer_a, answer_b, "Both_Correct_A_Default"
                
            verdict = self.judge_answers(user_content, answer_a, answer_b)
            
            if "A" in verdict:
                return answer_a, answer_b, "Both_Correct_Judge_A"
            elif "B" in verdict:
                return answer_b, answer_a, "Both_Correct_Judge_B"
            else:
                logger.warning("Judge could not decide")
                return None, None, "Both_Correct_Judge_Undecided"
                
        # Case 4: Both wrong
        else:
            return None, None, "Both_Wrong"
            
    def process_single_problem(self, row_idx: int, row: pd.Series) -> Optional[Dict]:
        """
        Process a single problem and generate DPO pair.
        
        Args:
            row_idx: Row index
            row: DataFrame row containing problem data
            
        Returns:
            DPO result dict or None if failed
        """
        try:
            # Parse problem
            problem = ast.literal_eval(row[self.cfg.data.columns.problems])
            gold = int(problem["answer"])
            
            # Format prompt
            user_content = format_problem_prompt(
                row[self.cfg.data.columns.paragraph],
                problem["question"],
                problem["choices"]
            )
            
            logger.info(f"[{row_idx}] Processing problem")
            
            # Generate answers from both models
            answer_a = self.generate_answer(user_content, self.cfg.model.model_a)
            answer_b = self.generate_answer(user_content, self.cfg.model.model_b)
            
            # Extract answer numbers
            ans_num_a = extract_answer_num(answer_a)
            ans_num_b = extract_answer_num(answer_b)
            
            if ans_num_a is None or ans_num_b is None:
                logger.warning(f"[{row_idx}] Failed to parse answers")
                return None
                
            # Select chosen/rejected
            chosen, rejected, strategy = self.select_chosen_rejected(
                answer_a, answer_b, ans_num_a, ans_num_b, gold, user_content
            )
            
            if chosen and rejected:
                return {
                    "prompt": user_content,
                    "chosen": chosen,
                    "rejected": rejected,
                    "gold": gold,
                    "strategy": strategy
                }
            else:
                logger.info(f"[{row_idx}] Skipped ({strategy})")
                return None
                
        except Exception as e:
            logger.error(f"[{row_idx}] Error: {e}")
            return None
            
    def generate_dataset(self, seed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate full DPO dataset from seed data.
        
        Args:
            seed_data: Input DataFrame with problems
            
        Returns:
            DataFrame with DPO pairs
        """
        results = []
        
        for idx in range(len(seed_data)):
            result = self.process_single_problem(idx, seed_data.iloc[idx])
            
            if result:
                results.append(result)
                logger.info(f"[{idx}] Collected ({result['strategy']}) | Total: {len(results)}")
                
            # Rate limiting
            time.sleep(self.cfg.processing.sleep_interval)
            
        return pd.DataFrame(results)
        
    def filter_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter dataset by strategy.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Filtered DataFrame
        """
        if self.cfg.strategy.filter_strategies:
            df = df[df['strategy'].isin(self.cfg.strategy.filter_strategies)]
            logger.info(f"Filtered to {len(df)} rows")
            
        return df
