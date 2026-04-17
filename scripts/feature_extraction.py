"""
Feature Extraction Script.
Extracts 39-dimensional MFCC features (MFCC, Delta, Delta-Delta) from audio chunks.
Saves the resulting feature matrix as a CSV file for model training.
"""
import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
# Set paths relative to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "Total_dataset")
SAVE_CSV_PATH = os.path.join(PROJECT_ROOT, "Output", "mfcc_features.csv")

# --- Execute ---
if __name__ == "__main__":
    results = []
    print("MFCC + Delta + Delta-Delta 특징 추출을 시작합니다...")
    print(f"입력 폴더: {INPUT_DIR}")
    print(f"출력 파일: {SAVE_CSV_PATH}")

    for label in ["snore", "nonsnore"]:
        folder_path = os.path.join(INPUT_DIR, label)
        
        if not os.path.isdir(folder_path):
            print(f"\n경고: '{folder_path}' 폴더를 찾을 수 없어 건너뜁니다.")
            continue
            
        files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
        
        if not files:
            print(f"경고: '{folder_path}' 폴더에 .wav 파일이 없습니다.")
            continue

        for file in tqdm(files, desc=f"Processing {label} files"):
            file_path = os.path.join(folder_path, file)

            try:
                y, sr = librosa.load(file_path, sr=16000, mono=True)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                delta = librosa.feature.delta(mfcc)
                delta2 = librosa.feature.delta(mfcc, order=2)

                feat = np.concatenate([mfcc, delta, delta2], axis=0)
                feat_mean = np.mean(feat, axis=1)

                results.append([file, label] + feat_mean.tolist())
            except Exception as e:
                print(f"오류 발생: {file} 처리 중 문제 발생 - {e}")

    if not results:
        print("\n오류: 처리된 오디오 파일이 없습니다. 입력 폴더 경로를 다시 확인해주세요.")
    else:
        columns = ["filename", "class"] + [f"feature_{i}" for i in range(39)]
        df = pd.DataFrame(results, columns=columns)

        save_dir = os.path.dirname(SAVE_CSV_PATH)
        os.makedirs(save_dir, exist_ok=True)
            
        df.to_csv(SAVE_CSV_PATH, index=False)

        print("\n-------------------------------------------")
        print(f"✅ 특징 추출 완료!")
        print(f"총 {len(df)}개의 샘플이 처리되었습니다.")
        print(f"결과가 '{SAVE_CSV_PATH}'에 저장되었습니다.")
        print("-------------------------------------------")
