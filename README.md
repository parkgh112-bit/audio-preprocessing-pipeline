# Audio Signal Processing & Feature Engineering Pipeline for Snoring Detection

본 리포지토리는 코골이 소리 감지 프로젝트에서 **데이터 엔지니어링 및 특징 추출(Feature Engineering) 파트**를 전담하여 구현한 결과물입니다. 원시 오디오 데이터를 딥러닝/머신러닝 모델이 학습 가능한 고차원 특징 벡터로 변환하는 전 과정을 최적화된 파이프라인으로 설계하였습니다.

## 🚀 핵심 구현 요약
 - 데이터 증강 및 정제: 1초 단위 정밀 슬라이싱($Slicing$)을 통해 학습 데이터셋을 약 5배 증강(9,800개 샘플 확보).특징 추출 최적화: $MFCC$ 및
 - $Delta/Delta$-$Delta$ 계수를 포함한 39차원 특징 벡터 추출 파이프라인 설계.
 - 모델 성능 기여: 정밀 전처리를 통해 최종 분류 모델의 Accuracy 95.6%, F1-Score 90.6% 달성 견인.
---

## 🛠 Tech Stack
- **Language**: Python 3.x
- **Libraries**: `Librosa` (Signal Processing), `NumPy` (Numerical Computing), `Soundfile` (I/O), `Pandas` (Data Management), `Tqdm` (Workflow Visualization)

---

## 📂 Pipeline Architecture
프로젝트는 모듈화된 설계를 통해 재사용성과 유지보수성을 확보했습니다.

```text
├── scripts/
│   ├── preprocess.py       # 오디오 슬라이싱 및 데이터 밸런싱
│   ├── feature_extraction.py # MFCC 특징 추출 및 CSV 저장
│   └── check_channels.py   # 채널 체크 및 오디오 유틸리티
└── Output/
    └── mfcc_features.csv   # 모델 학습용 최종 특징 행렬
```

---
## 데이터 수집

- Snoring (코골이) 데이터 정제:
  - HuggingFace(2,558개)와 Kaggle(500개) 데이터 중 중복된 500개 샘플을 발견.
  - 음질이 더 우수한 Kaggle 데이터를 우선 채택하여 최종 2,558개의 고품질 코골이 샘플 확보.
    
- Non-snoring (비코골이) 데이터 증강:
  - $ESC-50$ 환경음 데이터(5초 단위)를 1초 단위로 정밀 슬라이싱하여 9,800개의 샘플 확보.
  - Kaggle 비코골이 데이터 500개를 병합하여 총 10,300개의 네거티브 샘플 구축.
    
- Technical Decision: $ESC-50$의 코골이 데이터(40개)는 1초 분할 시 무음 구간 발생 및 라벨 노이즈(Label Noise) 위험으로 인해 최종 데이터셋에서 제외하여 모델의 학습 순도를 높임.

Snoring: 2,558개 | 
Non-snoring: 10,300개 | 
Total: 12,858개

---
## 🔍 Deep Dive: Engineering Details

### 1. Precision Slicing & Data Augmentation
데이터 불균형 문제를 해결하고 모델의 로컬 특징(Local Features) 학습 능력을 높이기 위해 슬라이싱 전략을 도입.
- **Problem**: 원본 데이터의 길이가 일정하지 않고, 학습에 가용한 샘플 수가 부족함.
- **Solution**: `scripts/preprocess.py`를 통해 5초 길이의 환경음을 1초 단위로 정밀 분할.
  근거: 코골이의 평균 호흡 주기($0.5s \sim 1.5s$)와 실시간 감지 저지연성($Low$-$latency$)을 고려하여 **1초($1s$)**를 표준 분석 단위로 설정.
  
- **Result**: 총 **9,800개의 유효 학습 샘플**을 확보, 데이터 부족으로 인한 오버피팅 문제를 선제적으로 차단했습니다.

### 2. Feature Engineering: 39-Dimensional Vector
단일 채널(Mono) 오디오에서 소리의 정적/동적 특성을 모두 포착하기 위해 39차원의 특징 벡터를 추출.

- MFCC (13-dim) - 정적 특징: 소리의 고유한 음색(Spectral Envelope) 포착.
- Delta (13-dim) - 동적 특징: 소리 특징의 시간당 변화율(속도) 계산. 단순 배경 소음과 코골이의 패턴 차이를 극대화.
- Delta-Delta (13-dim) - 동적 특징: 변화율의 변화량(가속도)을 계산하여 소리의 에너지 전이 과정을 정밀하게 묘사.

<img width="600" height="994" alt="image" src="https://github.com/user-attachments/assets/6cf8ec67-7cb6-436d-a7ed-46cd8f6afd01" />
<img width="600" height="1178" alt="image" src="https://github.com/user-attachments/assets/aa02a9a6-29cb-422f-9180-e1f772c24b8f" />
<img width="600" height="1167" alt="image" src="https://github.com/user-attachments/assets/9ba1be88-ea13-41c1-b6d2-24a1d45f649a" />


---

## 📈 Technical Contribution to Accuracy (95.6%)

본 전처리 파이프라인이 최종 모델 정확도 **95.6%** 달성에 기여한 핵심 이유는 다음과 같습니다.

1.  **Signal-to-Noise Ratio (SNR) 최적화**: 1초 단위 슬라이싱을 통해 모델이 배경 소음이 아닌 '코골이 이벤트' 자체에 더 집중(Focus)할 수 있는 짧은 윈도우를 제공.
2.  **Temporal Dynamics Capture**: Delta 및 Delta-Delta 특징을 결합하여 시계열 데이터의 전후 맥락을 벡터에 포함시킴으로써, 복잡한 환경음 속에서도 코골이 특유의 패턴을 강건하게(Robust) 분류가 가능.
3.  **Standardization**: 모든 오디오를 16kHz Mono 채널로 통일하고 고정 길이 벡터로 변환하여 모델 입력의 엔트로피를 최소화.

---

## 🏃 How to Run

### 1. Requirements
```bash
pip install -r requirements.txt
```

### 2. Slicing raw audio
```bash
python scripts/preprocess.py
```

### 3. Extracting features
```bash
python scripts/feature_extraction.py
```

---
**Note:** 이 리포지토리는 전체 프로젝트 중 데이터 엔지니어링 및 특징 추출(Feature Engineering) 파트를 중점적으로 다룹니다.
