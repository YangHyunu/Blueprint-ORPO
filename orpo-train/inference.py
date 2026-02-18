from unsloth import FastLanguageModel
import pandas as pd
import torch
import ast
# ==========================================
# 1. 설정 (학습 결과가 저장된 폴더 경로)
# 보통 'outputs' 폴더에 어댑터가 저장됩니다.
# ==========================================

Base_Prompt = """당신은 객관식 문제를 푸는 학생입니다.

문제를 해결할 때 다음 내부 절차를 반드시 따르세요:

1) 판단에 앞서, 이 문제와 직접적으로 관련된 핵심 사실·개념·원리를 먼저 정리한다.
2) 각 선지를 서로 독립적으로 검토하며, 해당 선지와 직접 연결되는 사실 또는 개념을 먼저 서술한다.
3) 개인의 행위·사례·시점이 아니라, 문제에서 요구하는 판단 대상(집단, 제도, 사상, 일반적 성격 등)을 기준으로 선지를 참/거짓으로 판정한다.
4) 모든 선지를 개별적으로 검증한 뒤, 오답을 제거하여 최종 정답을 확정한다.

※ 판단 과정에서 다음 오류를 피하라:
- 특정 인물의 행동으로 집단이나 사상의 성격을 단정하는 오류
- 특정 시점의 사례로 장기적·일반적 성격을 부정하는 오류
- 다른 선지와의 상대 비교로 참/거짓을 판단하는 오류

확실히 검증 가능한 사실과 개념에만 근거하여 판단하라.

출력 규칙을 반드시 지키세요.
[Answer]
반드시 맨 마지막 줄에만 아래 JSON을 출력:
{"Answer":"번호"}
"""

adapter_path = "outputs" 
data_path = "data/test.csv" 
# 2. 모델 로드 (Base Model + Adapter 합체)
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = adapter_path,
        max_seq_length = 4096,
        dtype = torch.bfloat16,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

except Exception as e:
    print("\n❌ 오류: 학습된 모델을 찾을 수 없습니다.")
    print("아직 학습(main.py)을 돌리지 않았거나, 'outputs' 폴더가 비어있습니다.")
    exit()

sample =pd.read_csv(data_path).iloc[0]
sample['problems'] = sample['problems'].apply(lambda x: ast.literal_eval(x),axis=1)


problem = sample['problems']
user_content = f"<제시문>\n{sample['paragraph']}\n\n<질문>\n{problem['question']}\n"
for k, choice in enumerate(problem['choices']):
    user_content += f"{k+1}. {choice}\n"


messages = [
    {"role": "system", "content": Base_Prompt},
    {"role": "user", "content": user_content},
]

print(f"\n질문 내용:\n{user_content}\n")

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")

# 5. 답변 생성
outputs = model.generate(
    inputs,
    max_new_tokens = 4096,      
    use_cache = True,
    temperature = 0.6,         
)

# 6. 결과 디코딩 및 출력
response = tokenizer.batch_decode(outputs)
print("\n" + "="*30)
# 프롬프트 부분을 제외하고 답변만 깔끔하게 출력하기 위해 파싱
print(response[0].split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip())
print("="*30 + "\n")