import unsloth 
from unsloth import PatchDPOTrainer

import hydra
from omegaconf import DictConfig
import wandb
import os
from dotenv import load_dotenv
from trl import ORPOConfig, ORPOTrainer

# 우리가 만든 모듈 임포트
from src.model import load_model_and_tokenizer
from src.data import load_and_format_dataset

load_dotenv()

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # 1. 실험 추적 시작 (WandB)
    wandb.init(project=cfg.wandb.project, name=cfg.wandb.run_name)
    
    # Unsloth 패치 적용 (메모리 최적화)
    PatchDPOTrainer()

    # 2. 모델 준비
    model, tokenizer = load_model_and_tokenizer(cfg)

    # 3. 데이터 준비 (유연한 로딩)
    dataset = load_and_format_dataset(cfg, tokenizer)
    
    # 데이터 확인용 출력
    print("="*30)
    print("Example Data:")
    print(dataset[0]['prompt'][:300]) # 앞부분만 출력 확인
    print("="*30)

    # 4. 학습 설정 (ORPO)
    orpo_args = ORPOConfig(
        per_device_train_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        max_length=cfg.training.max_seq_length,
        max_prompt_length=cfg.training.max_seq_length // 2,
        max_completion_length=cfg.training.max_seq_length // 2,
        num_train_epochs=cfg.training.num_train_epochs,
        logging_steps=cfg.training.logging_steps,
        output_dir=cfg.training.output_dir,
        optim=cfg.training.optim,
        warmup_ratio=cfg.training.warmup_ratio,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        beta= cfg.training.beta,
        report_to="wandb",
        remove_unused_columns=False, # 매핑된 데이터 보존을 위해 False 추천
    )

    trainer = ORPOTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=orpo_args,
    )

    # 5. 학습 시작
    print("🚀 Starting Training...")
    trainer.train()

    # 6. 저장
    final_path = os.path.join(cfg.training.output_dir, "final_model")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"✅ Model saved to {final_path}")

if __name__ == "__main__":
    main()