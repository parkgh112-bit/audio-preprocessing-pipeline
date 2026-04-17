"""
Audio Preprocessing Script for Snoring Detection Pipeline.
This script slices raw audio files into fixed-length chunks (e.g., 1-second)
to solve data imbalance and standardize input length for feature extraction.
"""
import os
import librosa
import soundfile as sf
from tqdm import tqdm
import numpy as np

def split_audio_files(input_folder, output_folder, chunk_duration=1, target_sr=16000):
    """
    Slices audio files in the specified folder into fixed-length chunks.

    :param input_folder: Path to the folder containing raw audio files.
    :param output_folder: Path to save the sliced audio files.
    :param chunk_duration: Duration of each chunk in seconds.
    :param target_sr: Target sampling rate.
    """
    os.makedirs(output_folder, exist_ok=True)
    print(f"결과가 '{output_folder}' 폴더에 저장됩니다.")

    if not os.path.exists(input_folder):
        print(f"오류: '{input_folder}' 폴더를 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    files_to_process = [f for f in os.listdir(input_folder) if f.endswith(".wav")]
    
    if not files_to_process:
        print(f"'{input_folder}'에 처리할 .wav 파일이 없습니다.")
        return

    for filename in tqdm(files_to_process, desc="오디오 파일 분할 중"):
        input_path = os.path.join(input_folder, filename)
        
        try:
            y, sr = librosa.load(input_path, sr=target_sr)
            chunk_length = chunk_duration * sr
            num_chunks = int(np.ceil(len(y) / chunk_length))

            for i in range(num_chunks):
                start_sample = i * chunk_length
                end_sample = start_sample + chunk_length
                y_chunk = y[start_sample:end_sample]

                if len(y_chunk) < chunk_length:
                    y_chunk = librosa.util.fix_length(y_chunk, size=chunk_length)

                base_filename = os.path.splitext(filename)[0]
                output_filename = f"{base_filename}_chunk_{i}.wav"
                output_path = os.path.join(output_folder, output_filename)

                sf.write(output_path, y_chunk, sr)

        except Exception as e:
            print(f"'{filename}' 처리 중 오류 발생: {e}")

    print("✅ 모든 파일 처리가 완료되었습니다.")

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DATA_FOLDER = os.path.join(PROJECT_ROOT, "data", "Non-snoring_ESC-50")
OUTPUT_DATA_FOLDER = os.path.join(PROJECT_ROOT, "data", "Non-snoring_ESC-50_1s")

# --- Execute ---
if __name__ == "__main__":
    split_audio_files(INPUT_DATA_FOLDER, OUTPUT_DATA_FOLDER)
