from unsloth import FastLanguageModel
import torch
def load_model_and_tokenizer(cfg):
    """설정(cfg)에 따라 모델과 토크나이저를 불러옵니다."""
    print(f"Loading Model: {cfg.model.path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.path,
        max_seq_length=cfg.training.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=cfg.model.load_in_4bit,
    )

    # LoRA 어댑터 부착
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.model.lora_r,
        target_modules=cfg.model.target_modules,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.training.seed,
    )
    
    return model, tokenizer