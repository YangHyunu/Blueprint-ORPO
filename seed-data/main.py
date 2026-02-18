"""Main script for DPO dataset generation with Hydra configuration."""

import os
import logging
from pathlib import Path
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf

from src.models.llm_client import LLMClient
from src.models.dpo_generator import DPODatasetGenerator


def setup_logging(cfg: DictConfig):
    """Setup logging configuration."""
    log_level = getattr(logging, cfg.logging.level)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if cfg.logging.save_logs:
        log_dir = Path(cfg.logging.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main function for DPO dataset generation.
    
    Args:
        cfg: Hydra configuration
    """
    # Setup logging
    setup_logging(cfg)
    logger = logging.getLogger(__name__)
    
    # Log configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Setup API key from environment
    api_key = os.environ.get(cfg.api.api_key_env)
    if not api_key:
        logger.warning(f"{cfg.api.api_key_env} not found in environment")
    
    # Initialize LLM client
    logger.info("Initializing LLM client...")
    llm_client = LLMClient(
        base_url=cfg.api.base_url,
        api_key=api_key,
        timeout=cfg.api.timeout
    )
    
    # Load seed data
    logger.info(f"Loading data from {cfg.data.input_path}...")
    seed_data = pd.read_csv(cfg.data.input_path)
    logger.info(f"Loaded {len(seed_data)} problems")
    
    # Initialize generator
    logger.info("Initializing DPO generator...")
    generator = DPODatasetGenerator(cfg, llm_client)
    
    # Generate dataset
    logger.info("Starting dataset generation...")
    df_dpo = generator.generate_dataset(seed_data)
    logger.info(f"Generated {len(df_dpo)} DPO pairs")
    
    # Apply filtering
    if cfg.strategy.filter_strategies:
        logger.info("Applying strategy filter...")
        df_dpo = generator.filter_dataset(df_dpo)
    
    # Save results
    output_path = Path(cfg.data.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_dpo.to_json(output_path, orient="records", lines=True, force_ascii=False)
    logger.info(f"Saved {len(df_dpo)} records to {output_path}")
    
    # Print statistics
    logger.info("\nStrategy distribution:")
    logger.info(df_dpo['strategy'].value_counts())
    
    logger.info("\n✅ Dataset generation completed successfully!")


if __name__ == "__main__":
    main()
