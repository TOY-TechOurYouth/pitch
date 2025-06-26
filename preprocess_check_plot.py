# visualize_features.py

import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
import random

# 한글 폰트 설정 (Windows 환경)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 📂 데이터 불러오기
features = joblib.load("files_for_train/features_175d.pkl")
labels = joblib.load("files_for_train/labels_gender.pkl")

features = np.array(features)
labels = np.array(labels)

# ✅ 1. 정규화 (Z-score or MinMax)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# ✅ 2. 라벨 인코딩
le = LabelEncoder()
label_encoded = le.fit_transform(labels)

# ✅ 3. PCA (전체 샘플 → 다운샘플링)
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

# ✅ 4. 무작위 1000개 샘플링
sample_size = 1000
random.seed(42)
sample_idx = random.sample(range(len(features_pca)), sample_size)
features_pca_sample = features_pca[sample_idx]
label_sample = label_encoded[sample_idx]

# ✅ 시각화 시작 (3행 1열 서브플롯)
fig, axs = plt.subplots(3, 1, figsize=(12, 18))

# ✅ 1. 히트맵 (정규화된 상위 50개 샘플)
sns.heatmap(features_scaled[:50], cmap="viridis", cbar=True, ax=axs[0])
axs[0].set_title("상위 50개 샘플의 정규화된 하이브리드 피처 히트맵")
axs[0].set_xlabel("피처 인덱스 (0~174)")
axs[0].set_ylabel("샘플 인덱스")

# ✅ 2. 평균/표준편차 그래프 (정규화 후라 비교 가능)
feature_mean = features_scaled.mean(axis=0)
feature_std = features_scaled.std(axis=0)

axs[1].plot(feature_mean, label="평균")
axs[1].plot(feature_std, label="표준편차")
axs[1].set_title("정규화된 175차원 피처의 평균 및 표준편차")
axs[1].set_xlabel("피처 인덱스")
axs[1].legend()
axs[1].grid(True)

# ✅ 3. PCA 분포 (1000개 샘플링 후 시각화)
scatter = axs[2].scatter(features_pca_sample[:, 0], features_pca_sample[:, 1],
                         c=label_sample, cmap="coolwarm", alpha=0.5)
handles = scatter.legend_elements()[0]
axs[2].legend(handles, le.classes_)
axs[2].set_title("PCA로 본 피처 분포 (2D, 샘플 1000개)")
axs[2].set_xlabel("PCA 1")
axs[2].set_ylabel("PCA 2")
axs[2].grid(True)

# ✅ 전체 저장
plt.tight_layout()
plt.savefig("feature_analysis_summary_fixed.png", dpi=300)
print("✅ 개선된 그래프가 feature_analysis_summary_fixed.png 로 저장되었습니다.")
