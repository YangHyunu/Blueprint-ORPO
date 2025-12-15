import unsloth
import torch
import transformers
from ast import literal_eval
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback
from datasets import Dataset
import json

import pandas as pd
import random
import numpy as np
import evaluate
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from peft import AutoPeftModelForCausalLM, LoraConfig

pd.set_option('display.max_columns', None)

# ============================================================
# 난수 고정
# ============================================================
def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

set_seed(42) # magic number :)

# ============================================================
# 모델 및 토크나이저 로드
# ============================================================
from unsloth import FastLanguageModel

max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# ============================================================
# Chat Template 설정
# ============================================================
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

print("Chat template:")
print(tokenizer.chat_template)

# ============================================================
# 데이터 로드 및 전처리
# ============================================================
dataset = pd.read_csv('./data/train.csv')

# Flatten the JSON dataset
records = []
for _, row in dataset.iterrows():
    problems = literal_eval(row['problems'])
    record = {
        'id': row['id'],
        'paragraph': row['paragraph'],
        'question': problems['question'],
        'choices': problems['choices'],
        'answer': problems.get('answer', None),
        "question_plus": problems.get('question_plus', None),
    }
    if 'question_plus' in problems:
        record['question_plus'] = problems['question_plus']
    records.append(record)
        
df = pd.DataFrame(records)

# ============================================================
# LoRA 설정
# ============================================================
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,
     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 64,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 42,
    use_rslora = False,
    loftq_config = None,
)

# ============================================================
# 프롬프트 템플릿
# ============================================================
PROMPT_NO_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""

PROMPT_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

<보기>:
{question_plus}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""

# ============================================================
# 데이터셋 변환
# ============================================================
dataset = Dataset.from_pandas(df)

processed_dataset = []
for i in range(len(dataset)):
    choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(dataset[i]["choices"])])

    if dataset[i]["question_plus"]:
        user_message = PROMPT_QUESTION_PLUS.format(
            paragraph=dataset[i]["paragraph"],
            question=dataset[i]["question"],
            question_plus=dataset[i]["question_plus"],
            choices=choices_string,
        )
    else:
        user_message = PROMPT_NO_QUESTION_PLUS.format(
            paragraph=dataset[i]["paragraph"],
            question=dataset[i]["question"],
            choices=choices_string,
        )

    processed_dataset.append(
        {
            "id": dataset[i]["id"],
            "messages": [
                {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": f"{dataset[i]['answer']}"}
            ],
            "label": dataset[i]["answer"],
        }
    )

processed_dataset = Dataset.from_pandas(pd.DataFrame(processed_dataset))

# ============================================================
# 토큰화 함수
# ============================================================
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["messages"])):
        output_texts.append(
            tokenizer.apply_chat_template(
                example["messages"][i],
                tokenize=False,
            )
        )
    return output_texts

def tokenize(element):
    outputs = tokenizer(
        formatting_prompts_func(element),
        truncation=False,
        padding=False,
        return_overflowing_tokens=False,
        return_length=False,
    )
    return {
        "input_ids": outputs["input_ids"],
        "attention_mask": outputs["attention_mask"],
    }

# 데이터 토큰화
tokenized_dataset = processed_dataset.map(
    tokenize,
    remove_columns=list(processed_dataset.features),
    batched=True,
    num_proc=4,
    load_from_cache_file=True,
    desc="Tokenizing",
)

# Train/Test split
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)

train_dataset = tokenized_dataset['train']
eval_dataset = tokenized_dataset['test']

print("Sample decoded input:")
print(tokenizer.decode(train_dataset[0]["input_ids"], skip_special_tokens=False))

# ============================================================
# 메트릭 설정
# ============================================================
acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
candidate_labels = ["1", "2", "3", "4", "5"]
int_output_map = {label: i for i, label in enumerate(candidate_labels)}

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple): 
        logits = logits[0]
    return logits.argmax(dim=-1)

def compute_metrics(eval_res):
    predictions, labels = eval_res
    
    # Numpy 변환
    if isinstance(predictions, torch.Tensor): 
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor): 
        labels = labels.cpu().numpy()

    final_preds, final_refs = [], []

    for i in range(len(labels)):
        # -100(패딩)이 아닌 유효한 인덱스만 추출
        valid_indices = np.where(labels[i] != -100)[0]
        if len(valid_indices) == 0: 
            continue
            
        # 정답 위치(target_idx) 찾기
        target_idx = valid_indices[-1]
        if labels[i][target_idx] == tokenizer.eos_token_id and len(valid_indices) > 1:
            target_idx = valid_indices[-2]
            
        # 예측 위치(pred_idx)는 정답 위치보다 한 칸 앞(-1)
        pred_idx = max(0, target_idx - 1)

        # 값 추출 및 디코딩
        decoded_label = tokenizer.decode([labels[i][target_idx]], skip_special_tokens=True).strip()
        decoded_pred = tokenizer.decode([predictions[i][pred_idx]], skip_special_tokens=True).strip()
        
        if i % 50 == 0:
            print(f"Decoded Label: {decoded_label}, Decoded Pred: {decoded_pred}")
        
        # 매핑 후 리스트 추가
        final_refs.append(int_output_map.get(decoded_label, -1))
        final_preds.append(int_output_map.get(decoded_pred, -1))

    return {
        "accuracy": acc_metric.compute(predictions=final_preds, references=final_refs)["accuracy"],
        "f1": f1_metric.compute(predictions=final_preds, references=final_refs, average="macro")["f1"]
    }

# ============================================================
# Pad token 설정
# ============================================================
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# ============================================================
# Trainer 설정
# ============================================================
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    compute_metrics=compute_metrics,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps = 1,
        warmup_ratio=0.1,
        num_train_epochs = 2,
        eval_steps=10,
        metric_for_best_model="eval_f1",
        eval_strategy="steps",
        save_strategy="best",
        save_steps=10,
        save_total_limit=1,
        learning_rate = 2e-5, 
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "cosine",
        seed = 42,
        load_best_model_at_end=True,
        report_to = "none",
        output_dir="./trainer_output",
    ),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# ============================================================
# Response-only 학습 설정
# ============================================================
from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)

# ============================================================
# GPU 메모리 확인
# ============================================================
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# ============================================================
# 학습 시작
# ============================================================
from unsloth import unsloth_train

if __name__ == "__main__":
    trainer_stats = unsloth_train(trainer)
    
    # 모델 저장
    print("Saving model...")
    model.save_pretrained("./final_model")
    tokenizer.save_pretrained("./final_model")
    print("Training completed and model saved!")
