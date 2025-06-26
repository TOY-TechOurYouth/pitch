import joblib
from sklearn.preprocessing import LabelEncoder

# 기존 pkl 파일에서 리스트 로드
class_list = joblib.load(r"/files_for_train/labels_gender.pkl")

# LabelEncoder 객체로 변환
le = LabelEncoder()
le.fit(class_list)

# 새롭게 저장
joblib.dump(le, "files_for_train/label_encoder.pkl")
print("✅ LabelEncoder 객체로 변환 완료 및 저장됨.")