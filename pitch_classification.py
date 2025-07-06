import os
import librosa
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# === 한글 폰트 설정 (Windows) ===
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# === 저장 디렉토리 설정 ===
RESULT_DIR = r"C:\Users\user\PycharmProjects\Toy2\pitch_result"
os.makedirs(RESULT_DIR, exist_ok=True)

# === 경로 설정 ===
CSV_PATH = r"C:\Users\user\PycharmProjects\TOY\gender_only\combined_pitch_data.csv"
AUDIO_DIRS = [
    r"C:\Users\user\PycharmProjects\TOY\four",
    r"C:\Users\user\PycharmProjects\TOY\five",
    r"C:\Users\user\PycharmProjects\TOY\five_2"
]
BEST_MODEL_PATH = r"C:\Users\user\PycharmProjects\Toy2\CNN_model_result\best_model.h5"
SCALER_PATH = r"C:\Users\user\PycharmProjects\Toy2\files_for_train\scaler.pkl"
ENCODER_PATH = r"C:\Users\user\PycharmProjects\Toy2\files_for_train\label_encoder.pkl"

SR = 16000 # 샘플링 레이트
DURATION = 0.25 # 1초 단위 청크
HYBRID_FEATURE_DIM = 175 # 추출될 feature 벡터의 길이

PITCH_THRESHOLDS = {
    'male': [85, 155, 180],
    'female': [165, 230, 300]
}

# === 함수 정의 ===

# wav_id에 해당하는 .wav 파일을 여러 폴더에서 찾아 경로를 반환
def find_wav_path(wav_id):
    for base_dir in AUDIO_DIRS:
        candidate = os.path.join(base_dir, wav_id + ".wav")
        if os.path.exists(candidate):
            return candidate, os.path.basename(base_dir)
    return None, None

# 오디오 파일에서 세개 피처 추출하여 1D 벡터로 연결
def extract_hybrid_feature(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        y, _ = librosa.effects.trim(y)
        if len(y) < int(sr * DURATION):
            y = np.pad(y, (0, int(sr * DURATION) - len(y)))
        else:
            y = y[:int(sr * DURATION)]

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        return np.concatenate([
            mfcc.mean(axis=1),
            mel.mean(axis=1),
            contrast.mean(axis=1)
        ])
    except Exception as e:
        print(f"⚠️ 오류 발생 (feature): {file_path} - {e}")
        return None

# librosa의 YIN 알고리즘을 이용해 평균 피치 추출
def extract_pitch(y, sr=16000):
    pitches = librosa.yin(y, fmin=50, fmax=500, sr=sr)
    pitches = pitches[~np.isnan(pitches)]
    return np.mean(pitches) if len(pitches) > 0 else None

# 평균 피치를 남/여 기준으로 분류
def classify_pitch(avg_pitch, gender):
    if gender not in PITCH_THRESHOLDS or avg_pitch is None:
        return "Unclear"
    low, mid, high = PITCH_THRESHOLDS[gender]
    if avg_pitch < mid:
        return "낮음"
    elif avg_pitch < high:
        return "중간"
    else:
        return "높음"

# CNN 모델 예측 결과의 confidence가 특정 threshold보다 낮으면 "unknown" 처리
def predict_with_uncertainty(model, sample_vector, label_encoder, threshold=0.75):
    if sample_vector.ndim == 1:
        sample_vector = sample_vector[..., np.newaxis]
    sample_vector = np.expand_dims(sample_vector, axis=0)
    probs = model.predict(sample_vector, verbose=0)[0]
    confidence = np.max(probs)
    predicted_index = np.argmax(probs)
    if confidence < threshold:
        return "unknown", probs
    else:
        return label_encoder.inverse_transform([predicted_index])[0], probs

# 전체 오디오를 1초 단위로 슬라이싱
def split_audio_to_chunks(y, sr, chunk_duration=1.0):
    chunk_len = int(sr * chunk_duration)
    return [y[i:i + chunk_len] for i in range(0, len(y), chunk_len) if len(y[i:i + chunk_len]) == chunk_len]

# === 데이터 로드 및 전처리 ===
df = pd.read_csv(CSV_PATH)
paths, sources = [], []
for wav_id in df["wav_id"]:
    path, source = find_wav_path(wav_id)
    paths.append(path)
    sources.append(source)

df["file_path"] = paths
df["source"] = sources
df = df.dropna(subset=["file_path"])

# === 모델 로드 ===
best_model = load_model(BEST_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# === 예측 및 저장 ===
results = []
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        file_path = row["file_path"]
        wav_id = row["wav_id"]
        source = row["source"]

        # [1] Feature
        feature_vector = extract_hybrid_feature(file_path)
        if feature_vector is None or feature_vector.shape[0] != HYBRID_FEATURE_DIM:
            raise ValueError("Invalid feature shape")

        # 스케일링(정규화)
        feature_scaled = scaler.transform([feature_vector])[0]

        # 회색지대 추출
        predicted_gender, probs = predict_with_uncertainty(best_model, feature_scaled, label_encoder)

        # 1초 단위 pitch 분류
        y, _ = librosa.load(file_path, sr=SR)
        y, _ = librosa.effects.trim(y)
        chunks = split_audio_to_chunks(y, sr=SR, chunk_duration=0.25)

        for idx, chunk in enumerate(chunks):
            pitch = extract_pitch(chunk, sr=SR)
            pitch_category = "중간" if predicted_gender == "unknown" else classify_pitch(pitch, predicted_gender)
            results.append((wav_id, source, idx + 1, predicted_gender, pitch, pitch_category))

    except Exception as e:
        results.append((row["wav_id"], "오류", -1, "오류", 0, "Unclear"))

# === 결과 저장 ===
final_df = pd.DataFrame(results, columns=["wav_id", "source", "chunk_index", "예측 성별", "평균 피치 (Hz)", "피치 카테고리"])
csv_path = os.path.join(RESULT_DIR, "final_gender_pitch_result_chunks_0,25.csv")
final_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"✅ 최종 CSV 저장 완료: {csv_path}")

# === 통계 출력 ===
print("\n📊 예측 성별 분포:")
print(final_df["예측 성별"].value_counts())
print("\n📊 피치 카테고리 분포:")
print(final_df["피치 카테고리"].value_counts())

# === 📊 시각화 1: 히트맵 ===
plt.figure(figsize=(8, 5))
cross_tab = pd.crosstab(final_df["예측 성별"], final_df["피치 카테고리"])
sns.heatmap(cross_tab, annot=True, fmt="d", cmap="Blues")
plt.title("성별 vs 피치 카테고리 분포 히트맵")
plt.xlabel("피치 카테고리")
plt.ylabel("예측 성별")
plt.tight_layout()
heatmap_path = os.path.join(RESULT_DIR, "heatmap_gender_pitch_0,25.png")
plt.savefig(heatmap_path)

# === 📊 시각화 2: 박스플롯 ===
plt.figure(figsize=(8, 5))
sns.boxplot(x="예측 성별", y="평균 피치 (Hz)", data=final_df)
plt.title("예측 성별별 평균 피치 박스플롯")
plt.ylabel("평균 피치 (Hz)")
plt.tight_layout()
boxplot_path = os.path.join(RESULT_DIR, "boxplot_pitch_by_gender_0.25.png")
plt.savefig(boxplot_path)
