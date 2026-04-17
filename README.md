# Audio Signal Processing & Feature Engineering Pipeline for Snoring Detection

본 리포지토리는 코골이 소리 감지 프로젝트에서 **데이터 엔지니어링 및 특징 추출(Feature Engineering) 파트**를 전담하여 구현한 결과물입니다. 원시 오디오 데이터를 딥러닝/머신러닝 모델이 학습 가능한 고차원 특징 벡터로 변환하는 전 과정을 최적화된 파이프라인으로 설계하였습니다.

## 🚀 핵심 구현 요약
- **Data Engineering**: 5초 단위의 원시 환경음(ESC-50)을 1초 단위로 정밀 슬라이싱하여 데이터 샘플을 9,800개로 대폭 확장 (Data Augmentation 효과).
- **Signal Processing**: Librosa와 NumPy를 활용하여 오디오 신호의 시계열 특성과 주파수 특성을 결합한 39차원 MFCC 추출 파이프라인 구축.
- **Performance Impact**: 정밀한 전처리와 특징 설계를 통해 최종 분류 모델의 **정확도를 95.6%까지 견인**.

---

## 🛠 Tech Stack
- **Language**: Python 3.x
- **Libraries**: `Librosa` (Signal Processing), `NumPy` (Numerical Computing), `Soundfile` (I/O), `Pandas` (Data Management), `Tqdm` (Workflow Visualization)

---

## 📂 Pipeline Architecture
프로젝트는 모듈화된 설계를 통해 재사용성과 유지보수성을 확보했습니다.

```text
├── data/
│   ├── raw/                # 원본 .wav 데이터 (ESC-50 등)
│   └── processed/          # 1초 단위로 슬라이싱된 데이터
├── scripts/
│   ├── preprocess.py       # 오디오 슬라이싱 및 데이터 밸런싱
│   ├── feature_extraction.py # MFCC 특징 추출 및 CSV 저장
│   └── check_channels.py   # 채널 체크 및 오디오 유틸리티
└── Output/
    └── mfcc_features.csv   # 모델 학습용 최종 특징 행렬
```

---

## 🔍 Deep Dive: Engineering Details

### 1. Precision Slicing & Data Augmentation
데이터 불균형 문제를 해결하고 모델의 로컬 특징(Local Features) 학습 능력을 높이기 위해 슬라이싱 전략을 도입했습니다.
- **Problem**: 원본 데이터의 길이가 일정하지 않고, 학습에 가용한 샘플 수가 부족함.
- **Solution**: `scripts/preprocess.py`를 통해 5초 길이의 환경음을 1초 단위로 정밀 분할. 
- **Result**: 마지막 조각의 손실을 방지하기 위해 **Padding 기법**을 적용하여 총 **9,800개의 유효 학습 샘플**을 확보, 데이터 부족으로 인한 오버피팅 문제를 선제적으로 차단했습니다.

### 2. 39-Dimensional MFCC Pipeline
인간의 청각 구조($Mel$-$scale$)를 모사하여 비선형 주파수 특성을 추출했습니다.
- **Static (13-dim)**: 음성 신호의 외포락(Spectral Envelope)을 나타내는 기본 MFCC.
- **Dynamic (26-dim)**: 시간 흐름에 따른 특징의 변화량인 **Delta**와 **Delta-Delta(Acceleration)** 계수를 추가 추출.
- **Technical Insight**: 코골이는 단순한 소음이 아니라 주기적인 파동의 변화를 가집니다. Delta 계수를 통해 소리의 '속도'와 '가속도' 변화를 캡처함으로써, 정적인 환경 소음과 동적인 코골이 소리를 명확히 구분해냈습니다.

---

## 📈 Technical Contribution to Accuracy (95.6%)

본 전처리 파이프라인이 최종 모델 정확도 **95.6%** 달성에 기여한 핵심 이유는 다음과 같습니다.

1.  **Signal-to-Noise Ratio (SNR) 최적화**: 1초 단위 슬라이싱을 통해 모델이 배경 소음이 아닌 '코골이 이벤트' 자체에 더 집중(Focus)할 수 있는 짧은 윈도우를 제공했습니다.
2.  **Temporal Dynamics Capture**: Delta 및 Delta-Delta 특징을 결합하여 시계열 데이터의 전후 맥락을 벡터에 포함시킴으로써, 복잡한 환경음 속에서도 코골이 특유의 패턴을 강건하게(Robust) 분류할 수 있게 했습니다.
3.  **Standardization**: 모든 오디오를 16kHz Mono 채널로 통일하고 고정 길이 벡터로 변환하여 모델 입력의 엔트로피를 최소화했습니다.

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
**Contact:** [Your Name/Email]
**Note:** 이 리포지토리는 전체 프로젝트 중 데이터 엔지니어링 및 특징 추출(Feature Engineering) 파트를 중점적으로 다룹니다.
