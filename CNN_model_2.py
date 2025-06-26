# train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# 한글 폰트 설정 (Windows 환경)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
X = joblib.load("files_for_train/features_175d.pkl")
y = joblib.load("files_for_train/labels_gender.pkl")
wav_ids = joblib.load("files_for_train/wav_ids.pkl")
sources = joblib.load("files_for_train/sources.pkl")  # 추가된 source 정보

X = np.array(X)
y = np.array(y)
wav_ids = np.array(wav_ids)
sources = np.array(sources)

# 다운샘플링
df = pd.DataFrame(X)
df['label'] = y
df['wav_id'] = wav_ids
df['source'] = sources

male_df = df[df['label'] == 'male']
female_df = df[df['label'] == 'female'].sample(n=len(male_df), random_state=42)
df_balanced = pd.concat([male_df, female_df]).sample(frac=1, random_state=42)

X_balanced = df_balanced.drop(columns=['label', 'wav_id', 'source']).values
y_balanced = df_balanced['label'].values
wav_ids_balanced = df_balanced['wav_id'].values
sources_balanced = df_balanced['source'].values

# 성별 분포 출력 및 시각화
gender_counts = Counter(y_balanced)
total = sum(gender_counts.values())
print("📊 학습에 사용된 성별 분포:")
for gender, count in gender_counts.items():
    print(f"{gender}: {count}명 ({count / total * 100:.2f}%)")

plt.figure(figsize=(6, 6))
plt.pie(gender_counts.values(), labels=gender_counts.keys(), autopct='%1.1f%%', startangle=90)
plt.title("학습 데이터 성별 분포")
plt.axis("equal")
plt.savefig("balanced_gender_distribution.png", dpi=300)
print("🖼️ 성별 분포 차트 saved as 'balanced_gender_distribution.png'")
plt.show()

# 정규화 및 저장
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_balanced)
joblib.dump(scaler, "files_for_train/scaler.pkl")

# 라벨 인코딩 및 One-hot
le = LabelEncoder()
y_encoded = le.fit_transform(y_balanced)
y_onehot = to_categorical(y_encoded)

# 데이터 분할
X_train, X_test, y_train, y_test, wav_train, wav_test, src_train, src_test = train_test_split(
    X_scaled, y_onehot, wav_ids_balanced, sources_balanced,
    test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\n📦 데이터 분할 결과:")
print(f"Train set: {X_train.shape[0]}개")
print(f"Test set : {X_test.shape[0]}개")

# 입력 차원 추가
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# CNN 모델 구성
model = Sequential([
    Conv1D(256, kernel_size=5, activation='relu', input_shape=(175, 1)),
    Conv1D(128, kernel_size=5, activation='relu'),
    Dropout(0.1),
    MaxPooling1D(pool_size=5),
    Conv1D(128, kernel_size=5, activation='relu'),
    Conv1D(128, kernel_size=5, activation='relu'),
    Flatten(),
    Dense(10, activation='relu'),
    Dense(y_onehot.shape[1], activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 콜백 정의
checkpoint = ModelCheckpoint(
    filepath='CNN_model_result/best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    mode='max',
    restore_best_weights=False,
    verbose=1
)

# 학습
model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[checkpoint, early_stop]
)

print("========================================================")

# 마지막 epoch 기준 모델 저장
model.save("gender_cnn_model_2.h5")
print("✅ 모델이 gender_cnn_model_2.h5 로 저장되었습니다.")

# 평가
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ [gender_cnn_model_2] 테스트 정확도: {acc * 100:.2f}%")

# 예측 수행
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test, axis=1)

# 혼동 행렬
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (gender_cnn_model_2)")
plt.savefig("gender_cnn_model_2_confusion_matrix.png", dpi=300)
plt.show()

print("\n📋 Classification Report (gender_cnn_model_2):")
print(classification_report(y_true, y_pred, target_names=le.classes_))

print("========================================================")

# === best_model 평가 ===
best_model = load_model("CNN_model_result/best_model.h5")
best_loss, best_acc = best_model.evaluate(X_test, y_test)
print(f"\n🌟 [best_model.h5] 기준 테스트 정확도: {best_acc * 100:.2f}%")

best_pred = best_model.predict(X_test)
best_y_pred = np.argmax(best_pred, axis=1)

cm_best = confusion_matrix(y_true, best_y_pred)
disp_best = ConfusionMatrixDisplay(confusion_matrix=cm_best, display_labels=le.classes_)
disp_best.plot(cmap='Oranges')
plt.title("Confusion Matrix (best_model.h5)")
plt.savefig("best_model_confusion_matrix.png", dpi=300)
plt.show()

print("\n📋 Classification Report (best_model.h5):")
print(classification_report(y_true, best_y_pred, target_names=le.classes_))

print("========================================================")

# 예측 결과 저장 (wav_id + source 포함)
results_df = pd.DataFrame({
    "wav_id": wav_test,
    "source": src_test,
    "true_label": le.inverse_transform(y_true),
    "predicted_label": le.inverse_transform(best_y_pred),
    "correct": y_true == best_y_pred
})

results_df.to_csv("gender_classification_results.csv", index=False, encoding="utf-8-sig")
print("📁 예측 결과가 'gender_classification_results.csv'로 저장되었습니다.")
