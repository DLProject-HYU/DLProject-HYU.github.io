# Title

# Members
이강후, 융합전자공학부, kanghoo7641@gmail.com   
김태은, 에너지공학과, qwertygkgk@naver.com   
김완준, 컴퓨터소프트웨어학부, wanjun226@gmail.com   
조연우, 기계공학부, s54s54@naver.com   

# Proposal
### 상수관로 진동 감지를 통한 누수 파악   
(Option 1)   
동기 : 상수관로에 설치된 자동 누수 감지 AI 모델을 통해 더욱 빠르고 범용적인 누수 감시 시스템의 필요   
기대 : 각 Hz대 별 진동 감지 데이터들을 입력받아 어느 유형의 누수가 발생했는지 파악하는 모델 생성   

***

# Datasets
장소에 따른 상수관로 누수음성 데이터셋(HZ)을 사용했습니다.
(Max(2n-2),Max(2n-1)) = (누수감지 n회 최대 주파수, 누수감지 n회 최대 누수 크기)
옥내누수, 옥외누수, 기계/전기음, 환경음, 정상음 5가지의 라벨이 붙어있었습니다.
우리가 알고자 하는건 누수가 발생했는지 아닌지 여부이기 때문에, 누수가 발생한 위치와 누수가 얼마나 발생했는지는 이용가치가 없는 데이터였습니다.
따라서, 이들을 제외하고 데이터를 다듬어 줄 필요가 있었기에, 예측에 필요없는 데이터인 site, sid, ldate, lrate, llevel을 제외했습니다.

# Methodology
Randomforest, xgboost를 사용했습니다.

# Evaluation & Analysis
실제 누수 2가지(in, out), 잘못된 누수 감지 2가지(noise, other), 정상음 1가지(normal)로 총 5개 라벨을 이용하여 Randomforst 모델을 활용하여 학습시켰습니다.
```
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
X = train.drop('leaktype', axis=1)
y = train['leaktype']

# 2. 데이터 분할 (훈련 데이터와 테스트 데이터로 분할)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 모델 학습
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# 4. 모델 평가
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```
5개 라벨 예측의 전체 정확도는 91.1%였습니다.

0~5120Hz 범위의 소리를 10Hz 단위로 측정하고, Max값 또한 20개를 포함한 데이터이다보니 컬럼의 수가 너무 많았습니다. 따라서 중요한 데이터를 찾아낼 필요가 있었습니다.

저음 영역대와 Max 값들의 중요도가 높음을 볼 수 있습니다.
따라서, 0~790Hz와 Max값들만을 사용하여 최적화를 진행했습니다.

전체 정확도 뿐만 아니라, 각각의 라벨들에 대한 정확도 모두 향상되었음을 확인할 수 있습니다.

예측 성능을 올리기 위해, Randomforest 모델 대신 XGBoost를 사용해봤습니다.

오히려 정확도가 낮아졌는데, 이는 XGBoost가 많은 하이퍼 파라미터를 가지고 있고, 이에 민감한 알고리즘 때문입니다.
따라서 이에 맞는 최적화가 필요하기 때문에 베이지안 최적화를 이용해봤습니다.


랜덤 포레스트 기본 모델 : Accuracy: 0.9114520898265803
랜덤 포레스트 피쳐 가공 : Accuracy: 0.9303923919124111
XGBoost 하이퍼파라미터 최적화 전 (피쳐 가공) : Accuracy: 0.8955486294253976
XGBoost 하이퍼파라미터 최적화 후 (피쳐 가공) : Accuracy: 0.9446175976983937

베이지언 최적화를 통해 하이퍼파라미터를 최적화 해주었을 경우 랜덤 포레스트 모델보다 높은 정확도를 보였습니다.

AIHub의 상관누수 데이터에서는 별도의 Test Data를 제공하기 때문에, 랜덤 포레스트 기본모델, 랜덤포레스트 피쳐 가공 모델, XGBoost 피쳐 가공 모델, XGBoost 피쳐 가공 + 하이퍼파라미터 튜닝 모델 4가지의 성능을 비교해봤습니다.



# Related Work

# Conclusion: Discussion
