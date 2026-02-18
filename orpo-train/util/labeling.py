import pandas as pd
from ast import literal_eval
from openai import OpenAI
import os
from dotenv import load_dotenv
from tqdm import tqdm


dataset = pd.read_csv('data/train.csv') 

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
    # Include 'question_plus' if it exists
    if 'question_plus' in problems:
        record['question_plus'] = problems['question_plus']
    records.append(record)
        
df = pd.DataFrame(records)

CATEGORIES = [
    "한국사", "역사", "경제", "정치", "지리",
    "심리", "교육산업", "국제", "부동산",
    "사회", "생활", "책마을"
]

def build_prompt(paragraph: str, question: str) -> str:
    categories = ", ".join(CATEGORIES)
    return f"""
다음은 수능 스타일의 문제이다.

[지문]
{paragraph}

[문제]
{question}

이 문제의 과목을 다음 목록 중 하나로 분류하라.
목록: [{categories}]

규칙:
- 반드시 목록 중 하나만 정확히 출력할 것
- 설명하지 말고 과목명만 출력할 것
"""

def classify_category(client, paragraph: str, question: str) -> str:
    prompt = build_prompt(paragraph, question)

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt,
        reasoning={"effort": "low"},
        text={"verbosity": "low"}
    )

    # 모델 출력 텍스트 추출
    category = response.output_text.strip()

    # 안전장치: 허용 목록 외 출력 방지
    if category not in CATEGORIES:
        return "UNKNOWN"

    return category

def main():
    load_dotenv('.env')  
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    tqdm.pandas()
    df["category"] = df.progress_apply(
        lambda row: classify_category(
            client,
            paragraph=row["paragraph"],
            question=row["question"]
        ),
        axis=1
    )
    
    output_path = "data/train_with_category.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
if __name__ == "__main__":
    main()
    
    