import pandas as pd
from collections import Counter

# CSV 파일 불러오기
df = pd.read_csv(r"C:\Users\user\PycharmProjects\TOY\raw_data\four.csv", encoding='cp949')

# 필요한 열만 추출 (wav_id와 성별 열, 예: '성별'이라는 컬럼명으로 존재한다고 가정)
result = df[['wav_id', '성별']]

# 출력
print(result)

# 저장
result.to_csv("gender_only_four.csv", index=False, encoding='utf-8-sig')