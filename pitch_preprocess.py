import librosa
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

save_dir=r"C:\Users\user\PycharmProjects\TOY\pitch_plot"
os.makedirs(save_dir, exist_ok=True)  # 폴더 없으면 생성

# 실제 wav 파일들이 있는 디렉토리 경로
base_audio_dir = r"C:\Users\user\PycharmProjects\TOY\four"

# CSV에서 wav_id와 label(성별) 불러오기
df = pd.read_csv(r"/gender_only/gender_only_four.csv")


# 파일 경로 생성
df["file_path"] = df["wav_id"].apply(lambda x: os.path.join(base_audio_dir, x + ".wav"))
file_paths = df["file_path"].values
labels = df["성별"].values  # 여기서 label = 성별(M/F)

# 성별 그룹별 피치 기준값 (Hz)
PITCH_THRESHOLDS = {
    'male': [85, 155, 180],    # 남성: 낮음 < 155 < 중간 < 180 < 높음
    'female': [165, 230, 300],   # 여성: 낮음 < 230 < 중간 < 300 < 높음
}

import os

missing_files = []
for filepath in file_paths:
    if not os.path.exists(filepath):
        missing_files.append(filepath)

print(f"존재하지 않는 파일 수: {len(missing_files)}")
for mf in missing_files:
    print("❌ 없음:", mf)


# 피치 추출 및 분류 함수
def extract_pitch_and_classify(filepath, group):
    y, sr = librosa.load(filepath)
    pitches = librosa.yin(y, fmin=50, fmax=500, sr=sr)
    pitches = pitches[~np.isnan(pitches)]

    if len(pitches) == 0 or group not in PITCH_THRESHOLDS:
        print(f"[경고] 피치 없음: {filepath}")
        return None, "Unclear"

    avg_pitch = np.mean(pitches)
    low, mid, high = PITCH_THRESHOLDS[group]

    if avg_pitch < mid:
        category = "낮음"
    elif avg_pitch < high:
        category = "중간"
    else:
        category = "높음"

    return avg_pitch, category

# 피치 분석 실행
results = []
for filepath, group in zip(file_paths, labels):
    filename = os.path.basename(filepath)
    pitch, category = extract_pitch_and_classify(filepath, group)
    if pitch is not None:
        results.append((filename, group, pitch, category))
    else:
        results.append((filename, group, 0, category))

# 결과 CSV로 저장
with open('pitch_gender_results_four.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['파일명', '성별', '평균 피치 (Hz)', '카테고리'])
    writer.writerows(results)

# 시각화
pitches = [row[2] for row in results]
categories = [row[3] for row in results]

color_map = {
    "낮음": "skyblue",
    "중간": "orange",
    "높음": "salmon",
    "Unclear": "gray"
}
colors = [color_map.get(cat, "gray") for cat in categories]


x_indices = list(range(len(pitches)))

plt.figure(figsize=(12, 6))
plt.bar(x_indices, pitches, color=colors)
plt.xticks(rotation=45, ha='right')
plt.xlabel('파일 인덱스')
plt.ylabel('평균 피치 (Hz)')
plt.title('남성/여성 음성 파일의 평균 피치 분류 결과')
plt.tight_layout()

# 파일 이름 설정 후 저장
save_path = os.path.join(save_dir, f"pitch_plot_four_second_6.png")
plt.savefig(save_path)
plt.close()