import pandas as pd
from datasets import load_dataset, Dataset
import os

def load_and_format_dataset(cfg, tokenizer):
    """
    설정(cfg)에 따라 로컬 파일 혹은 허깅페이스 데이터를 로드하고,
    컬럼 이름을 동적으로 매핑하여 포맷팅합니다.
    """
    data_path = cfg.dataset.path
    print(f"Loading Dataset from: {data_path}")

    # 1. 데이터 소스 판별 및 로드
    if data_path.endswith('.csv'):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {data_path}")
        df = pd.read_csv(data_path)
        dataset = Dataset.from_pandas(df)
    else:
        # 허깅페이스 허브에서 로드
        dataset = load_dataset(data_path, split="train")

    # 2. 컬럼 이름 가져오기 (설정에 없으면 기본값 'prompt', 'chosen', 'rejected' 사용)
    # getattr(객체, 속성명, 기본값)
    c_prompt = getattr(cfg.dataset, "col_prompt", "prompt")
    c_chosen = getattr(cfg.dataset, "col_chosen", "chosen")
    c_rejected = getattr(cfg.dataset, "col_rejected", "rejected")

    print(f"Using columns mapping -> Prompt: {c_prompt}, Chosen: {c_chosen}, Rejected: {c_rejected}")

    # 3. 포맷팅 함수
    def formatting_prompts_func(example):

        prompt_text = example[c_prompt]
        
        # 시스템 프롬프트 적용
        messages = [
            {"role": "system", "content": cfg.dataset.system_prompt},
            {"role": "user", "content": prompt_text}
        ]
        
        # 채팅 템플릿 적용 (텍스트로 변환)
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ) 
        

        return {
            "prompt": formatted_prompt,
            "chosen": example[c_chosen],
            "rejected": example[c_rejected]
        }

    # 4. 매핑 적용
    dataset = dataset.map(formatting_prompts_func)
    return dataset