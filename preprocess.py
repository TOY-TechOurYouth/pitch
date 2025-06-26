import librosa
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
import joblib  # 피처 저장용
from tqdm import tqdm

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# CSV 불러오기
df = pd.read_csv(r"C:\Users\user\PycharmProjects\TOY\gender_only\combined_pitch_data.csv")

# 디렉토리 설정
audio_dirs = [
    r"C:\Users\user\PycharmProjects\TOY\four",
    r"C:\Users\user\PycharmProjects\TOY\five",
    r"C:\Users\user\PycharmProjects\TOY\five_2"
]

# 각 디렉토리에서 파일 경로 찾기
def find_wav_path(wav_id):
    for base_dir in audio_dirs:
        candidate = os.path.join(base_dir, wav_id + ".wav")
        if os.path.exists(candidate):
            source = os.path.basename(base_dir)  # 디렉토리 이름
            return candidate, source
    return None, None

# 경로 및 소스 정보 추가
paths, sources = [], []
for wav_id in df["wav_id"]:
    path, source = find_wav_path(wav_id)
    paths.append(path)
    sources.append(source)

df["file_path"] = paths
df["source"] = sources
df = df.dropna(subset=["file_path"])  # 경로 못 찾은 파일 제거

# 피처 추출 함수
def extract_hybrid_feature(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        # 시간축 평균
        return np.concatenate([
            mfcc.mean(axis=1),
            mel.mean(axis=1),
            contrast.mean(axis=1)
        ])
    except Exception as e:
        print(f"오류 발생: {file_path}, {e}")
        return None


# 피처 추출
features, labels, wav_ids, source_list = [], [], [], []

for i, row in tqdm(df.iterrows(), total=len(df)):
    f = extract_hybrid_feature(row["file_path"])
    if f is not None and f.shape[0] == 175:
        features.append(f)
        labels.append(row["성별"])
        wav_ids.append(row["wav_id"])
        source_list.append(row["source"])


# 저장
joblib.dump(features, "files_for_train/features_175d.pkl")
joblib.dump(labels, "files_for_train/labels_gender.pkl")
joblib.dump(wav_ids, "files_for_train/wav_ids.pkl")
joblib.dump(source_list, "files_for_train/sources.pkl")

print("🎉 피처와 레이블 저장 완료!")