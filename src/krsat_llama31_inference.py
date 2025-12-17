"""
KRSAT Llama 3.1 Inference Script
수능 문제 풀이를 위한 Llama 3.1 모델 추론 스크립트

⚠️ 시스템 요구사항:
- 최소 8GB 이상의 디스크 여유 공간 (모델 다운로드용)
- 최소 16GB RAM 권장
- GPU 사용 시: CUDA 지원 GPU 권장 
"""


import re
from ast import literal_eval
import pandas as pd
import random
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from llama_cpp import Llama


# 프롬프트 템플릿 정의
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


def load_and_prepare_test_data(test_csv_path):
    """테스트 데이터를 로드하고 전처리합니다."""
    print(f"테스트 데이터 로딩 중: {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)
    
    # Flatten the JSON dataset
    records = []
    for _, row in test_df.iterrows():
        problems = literal_eval(row['problems'])
        record = {
            'id': row['id'],
            'paragraph': row['paragraph'],
            'question': problems['question'],
            'choices': problems['choices'],
            'answer': problems.get('answer', None),
            "question_plus": problems.get('question_plus', None),
        }
        # Include 'question_plus' if it exists
        if 'question_plus' in problems:
            record['question_plus'] = problems['question_plus']
        records.append(record)
    
    # Convert to DataFrame
    test_df = pd.DataFrame(records)
    print(f"{len(test_df)}개의 테스트 샘플 로드 완료")
    return test_df


def prepare_test_dataset(test_df):
    """테스트 데이터셋을 모델 입력 형식으로 변환합니다."""
    print("테스트 데이터셋 변환 중...")
    test_dataset = []
    
    for i, row in test_df.iterrows():
        choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])
        len_choices = len(row["choices"])

        # <보기>가 있을 때
        if row["question_plus"]:
            user_message = PROMPT_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                question_plus=row["question_plus"],
                choices=choices_string,
            )
        # <보기>가 없을 때
        else:
            user_message = PROMPT_NO_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                choices=choices_string,
            )

        test_dataset.append(
            {
                "id": row["id"],
                "messages": [
                    {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                    {"role": "user", "content": user_message},
                ],
                "label": row["answer"],
                "len_choices": len_choices,
            }
        )
    
    print(f"{len(test_dataset)}개의 샘플 변환 완료")
    return test_dataset


def download_model(repo_id, filename):
    """HuggingFace Hub에서 모델을 다운로드합니다."""
    print(f"모델 다운로드 중: {repo_id}/{filename}")
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
    )
    print(f"다운로드 완료: {model_path}")
    return model_path


def load_llm_model(model_path, n_ctx=4096, n_gpu_layers=-1):
    """Llama 모델을 로드합니다."""
    print(f" 모델 로딩 중...")
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    print(f"모델 로드 완료")
    return llm


def run_inference(llm, test_dataset, max_tokens=10, temperature=0.1, top_p=0.9):
    """테스트 데이터셋에 대해 추론을 실행합니다."""
    print(f"🔮 추론 시작 (총 {len(test_dataset)}개 샘플)...")
    infer_results = []
    
    for data in tqdm(test_dataset, desc="Inference"):
        _id = data["id"]
        messages = data["messages"]

        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        # 응답 텍스트 추출
        out_text = response['choices'][0]['message']['content'].strip()

        # 1~5 사이의 숫자 추출 (정답 형식이 "1", "1.", "정답: 1" 등으로 나올 수 있음)
        match = re.search(r'[1-5]', out_text)
        if match:
            predict_value = match.group(0)
        else:
            # 숫자를 찾지 못한 경우 (찍어)
            print(f"경고: 샘플 ID {_id}에서 유효한 답변을 찾지 못했습니다. 출력: '{out_text}'")
            
            predict_value = str(random.randint(1, min(data["len_choices"], 5)))

        infer_results.append({"id": _id, "answer": predict_value})
    
    print(f"추론 완료")
    return infer_results


def save_results(infer_results, output_path):
    """추론 결과를 CSV 파일로 저장합니다."""
    print(f"결과 저장 중: {output_path}")
    pred_df = pd.DataFrame(infer_results)
    pred_df.to_csv(output_path, index=False)
    print(f"저장 완료")
    print(f"\n결과 미리보기:")
    print(pred_df.head())
    return pred_df


def main():
    """메인 실행 함수"""
    # === 설정 ===
    TEST_CSV_PATH = './data/test.csv'  # 테스트 데이터 경로
    OUTPUT_PATH = 'krsat_predictions.csv'  # 출력 파일 경로
    
    MODEL_REPO_ID = "Hyunwoo98/Llama-3.1-8B-KRSAT-GGUF"
    MODEL_FILENAME = "Meta-Llama-3.1-8B-Instruct.Q8_0.gguf"
    
    # === 데이터 로드 및 전처리 ===
    test_df = load_and_prepare_test_data(TEST_CSV_PATH)
    test_dataset = prepare_test_dataset(test_df)
    
    # === 모델 다운로드 및 로드 ===
    model_path = download_model(MODEL_REPO_ID, MODEL_FILENAME)
    llm = load_llm_model(model_path)
    
    # === 샘플 테스트 (선택사항) ===
    print("\n 샘플 테스트 실행...")
    sample_data = test_dataset[0]
    print(f"질문 미리보기: {sample_data['messages'][1]['content'][:]}...")
    
    response = llm.create_chat_completion(
        messages=sample_data['messages'],
        max_tokens=10,
        temperature=0.2,
    )
    
    print("\n--- 모델 응답 ---")
    print(response['choices'][0]['message']['content'])
    print("---------------\n")
    
    # === 전체 추론 실행 ===
    infer_results = run_inference(llm, test_dataset)
    
    # === 결과 저장 ===
    pred_df = save_results(infer_results, OUTPUT_PATH)
    
    print(f"\n모든 작업이 완료되었습니다!")
    print(f"총 {len(infer_results)}개의 예측 결과가 '{OUTPUT_PATH}'에 저장되었습니다.")


if __name__ == "__main__":
    main()
