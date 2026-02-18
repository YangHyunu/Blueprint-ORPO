
# Unsloth + Hydra based ORPO Fine-tuning Project

## 📂 Project Structure

```bash
Unit/
├── configs/                    # [Config] All training configurations
│   ├── config.yaml             # Main configuration file
│   ├── model/                  # Model specific settings (path, LoRA)
│   │   └── Qwen3-14B-bnb-4bit.yaml
│   └── dataset/                # Dataset specific settings
│       └── Hard_Dataset_Orpo.yaml
│
├── src/                        # [Source] Core training code
│   ├── main.py                 # Training entry point script
│   └── model.py                # Model loading & Unsloth logic
│
├── outputs/                    # [Results] Saved LoRA adapters (Ignored by Git)
│   └── final_model/
│
├── inference.py                # [Inference] Testing script
├── .env.example                # [Security] Env variable template
├── .gitignore                  # [Git] Ignored file list
├── pyproject.toml              # [Dependency] uv package manager file
└── README.md                   # [Docs] Project documentation
```

이 프로젝트는 **Unsloth(가속 라이브러리)**와 **Hydra(설정 관리)**를 결합하여, LLM을 효율적으로 **ORPO(Odds Ratio Preference Optimization)** 방식으로 파인튜닝하기 위한 템플릿입니다.

팀원 누구나 설정을 쉽게 변경하고 실험을 재현할 수 있도록 구조화되었습니다.

---

## 1. 설치 가이드 (Installation)

이 프로젝트는 라이브러리 간의 버전 민감도가 높습니다. **반드시 `uv`를 사용하여 아래 명령어로 설치해주세요.** (Colab 최신 환경과 로컬 환경 동기화)

```bash
# 프로젝트 루트(Unit)에서 실행
cd Unit

# 의존성 충돌 해결 및 필수 패키지 설치
uv add "datasets<4.4.0" "trl>=0.19.0" "unsloth-zoo @ git+[https://github.com/unslothai/unsloth-zoo.git](https://github.com/unslothai/unsloth-zoo.git)" "unsloth[colab-new] @ git+[https://github.com/unslothai/unsloth.git](https://github.com/unslothai/unsloth.git)"

```

---

## 2. 설정 파일 상세 설명 (Configuration)

모든 학습 설정은 `configs/` 폴더 내의 `.yaml` 파일로 관리됩니다.

### 2.1 메인 설정 (`configs/config.yaml`)

전체 학습의 **컨트롤 타워** 역할을 합니다. 어떤 모델과 데이터셋을 조립할지 결정합니다.

```yaml
defaults:
  - model: Qwen3-14B-bnb-4bit       # configs/model/ 폴더 내 파일 선택
  - dataset: Hard_Dataset_Orpo # configs/dataset/ 폴더 내 파일 선택
  - _self_

training:
  output_dir: "outputs"       # 결과 저장 경로
  num_train_epochs: 2         # 학습 반복 횟수 (데이터 전체를 1번 훓음)
  max_seq_length: 8192        # 입력 시퀀스 최대 길이 
  
  # [중요] ORPO 핵심 파라미터
  beta: 0.15                   # ORPO Beta 값 (Rejected 답변에 대한 페널티 강도)
                              # 보통 0.1 ~ 0.3 사용. 너무 크면 언어 능력이 망가짐.

  # 학습 속도 및 메모리 관련
  batch_size: 1               # 한 번에 GPU에 올릴 데이터 수
  gradient_accumulation_steps: 6 # 배치를 모아서 업데이트 (실제 배치 = 1 * 6 = 6)
  learning_rate: 5e-6         
  optim: "adamw_8bit"         # 8bit 옵티마이저 (메모리 절약)

```

### 2.2 모델 설정 (`configs/model/*.yaml`)

모델의 경로와 **LoRA(Adapter)** 설정을 관리합니다.

```yaml
name: "Qwen3-14B-bnb-4bit"
path: "unsloth/Qwen3-14B-Instruct-bnb-4bit" # HuggingFace 모델 ID
load_in_4bit: true             # 4bit 양자화 로드 (메모리 4배 절약)

# [LoRA 파라미터 설명]
lora_r: 16                     # Rank: 어댑터의 크기 (높을수록 똑똑하지만 무거움, 보통 8~64)
lora_alpha: 16                 # Alpha: 학습 반영 비율 (보통 r과 같게 하거나 2배로 설정)
lora_dropout: 0                # 0 권장 (Unsloth 최적화 기능)

# 학습시킬 모듈 # mlp 레이어만 학습
target_modules: ["gate_proj", "up_proj", "down_proj"]
chat_template: "qwen3-thinking"  

```

---

## 3. 실행 방법 (Usage)

### 기본 학습 실행

`config.yaml`에 적힌 기본값으로 학습을 시작합니다.

```bash
uv run main.py

```

### 실험용: 설정 덮어쓰기 (Overrides)

파일을 수정하지 않고 명령어만으로 설정을 바꿔서 실험할 수 있습니다.

```bash
# 예: 데이터셋을 바꾸고 에폭을 3으로 늘려서 실행
uv run main.py dataset=korean_history training.num_train_epochs=3
```

---
