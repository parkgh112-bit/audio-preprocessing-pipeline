"""
Audio Channel Checker.
Utility script to verify the number of audio channels (mono/stereo) in a given directory
to ensure data consistency before feature extraction.
"""
import os
import librosa
from tqdm import tqdm

def check_audio_channels_in_folder(folder_path):
    """
    Checks and summarizes the number of audio channels for all .wav files in a folder.

    :param folder_path: Path to the folder containing audio files.
    """
    if not os.path.isdir(folder_path):
        print(f"오류: '{folder_path}'는 유효한 폴더가 아닙니다.")
        return

    wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    
    if not wav_files:
        print(f"'{folder_path}' 폴더에 .wav 파일이 없습니다.")
        return

    channel_counts = {}
    total_files = len(wav_files)
    
    print(f"총 {total_files}개의 .wav 파일을 검사합니다...")

    for filename in tqdm(wav_files, desc="파일 처리 중"):
        file_path = os.path.join(folder_path, filename)
        
        try:
            y, sr = librosa.load(file_path, sr=None, mono=False)
            
            if y.ndim == 1:
                channels = 1
            else:
                channels = y.shape[0]
            
            if channels in channel_counts:
                channel_counts[channels] += 1
            else:
                channel_counts[channels] = 1

        except Exception as e:
            print(f"\n'{filename}' 파일 처리 중 오류 발생: {e}")

    print("\n--- 검사 결과 요약 ---")
    for channels, count in channel_counts.items():
        print(f"채널 수 {channels}: {count}개 파일")
    print("--------------------")

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDER_TO_CHECK = os.path.join(PROJECT_ROOT, "data", "허깅페이스 snore")

# --- Execute ---
if __name__ == "__main__":
    check_audio_channels_in_folder(FOLDER_TO_CHECK)
