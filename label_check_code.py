import joblib
from collections import Counter
import matplotlib.pyplot as plt

# 한글 폰트 설정 (Windows 환경)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. 성별 라벨 불러오기
labels = joblib.load(r"/files_for_train/labels_gender.pkl")  # "남자", "여자" 등

# 2. 성별 비율 계산
gender_counts = Counter(labels)
total = sum(gender_counts.values())

print("📊 성별 분포:")
for gender, count in gender_counts.items():
    print(f"{gender}: {count}명 ({count / total * 100:.2f}%)")

# 3. 파이 차트 시각화
plt.figure(figsize=(6, 6))
plt.pie(gender_counts.values(), labels=gender_counts.keys(), autopct='%1.1f%%', startangle=90)
plt.title("성별 비율")
plt.axis("equal")  # 원형 유지

# 4. 차트 저장
plt.savefig("gender_distribution.png", dpi=300)
print("✅ 파이 차트가 'gender_distribution.png'로 저장되었습니다.")
