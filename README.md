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
장소에 따른 상수관로 누수음성 데이터셋(HZ)  
옥내누수, 옥외누수, 기계/전기음, 환경음, 정상음 포함
test와 train셋으로 분리함. Test,Train은 xgboost를 활용한 코드에 쓰인 데이터 셋, test, training은 순수 randomforest을 이용한 코드에 사용되었습니다.
# Methodology
Randomforest, xgboost
# Evaluation & Analysis
##  **1. 랜덤 포레스트**
랜덤 포레스트 알고리즘에는 세 개의 주요 하이퍼파라미터가 있습니다. 훈련 전에 이러한 하이퍼파라미터를 설정해야 합니다. 여기에는 노드 크기, 트리의 수, 샘플링된 특성의 수가 포함됩니다. 여기에서부터 랜덤 포레스트 분류자를 사용하여 회귀 또는 분류 문제를 해결할 수 있습니다.
랜덤 포레스트 알고리즘은 다수의 의사결정 트리로 구성되며 앙상블의 각 트리는 복원 추출 방식으로 훈련 세트에서 추출된 데이터 샘플(이를 부트스트랩 샘플이라고 부름)로 구성됩니다. 이 훈련 샘플 중에서 1/3은 테스트 데이터로 떼어 놓습니다. 그 다음, 무작위성의 또 다른 인스턴스는 특성 배깅을 통해 주입됩니다. 이를 통해 데이터 세트에 다양성을 추가하고 의사결정 트리 간의 상관관계를 줄입니다. 문제의 유형에 따라 예측에 대한 결정이 달라집니다. 회귀 작업의 경우 개별 의사결정 트리는 평균을 구하며, 분류 작업의 경우 다수결 보트(majority vote), 즉 가장 빈번한 범주적 변수에 따라 예측된 클래스를 내놓습니다. 마지막으로, oob 샘플이 교차 검증을 위해 사용되고 예측이 완료됩니다.
###   **1.1. 랜덤포레스트 기본 모델**
실제 누수 2가지(in, out), 잘못된 누수 감지 2가지(noise, other), 정상음 1가지(normal)로 총 5개 라벨을 이용하여 Randomforst 모델을 활용하여 학습시켰습니다.
```
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
X = train.drop('leaktype', axis=1)
y = train['leaktype']
```
X = train.drop('leaktype', axis=1): 데이터프레임 train에서 'leaktype' 열을 제외한 모든 열을 특성(X)으로 사용합니다.
y = train['leaktype']: 'leaktype' 열을 라벨(y)로 사용합니다.
```
# 2. 데이터 분할 (훈련 데이터와 테스트 데이터로 분할)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
train_test_split 함수를 사용하여 데이터를 훈련 데이터(X_train, y_train)와 테스트 데이터(X_test, y_test)로 나눕니다. 이때, 테스트 데이터의 비율은 20%로 지정되어 있습니다.
```
# 3. 모델 학습
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)
```
RandomForestClassifier를 사용하여 랜덤 포레스트 분류기 모델을 생성합니다.
fit 메서드를 사용하여 훈련 데이터를 사용하여 모델을 학습시킵니다.
```
# 4. 모델 평가
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```
```
Accuracy: 0.9114520898265803
```
```
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```
```
              precision    recall  f1-score   support

          in       0.91      0.87      0.89      2622
       noise       0.84      0.77      0.81       988
      normal       0.95      1.00      0.97      3933
       other       0.94      0.79      0.86      1481
         out       0.88      0.94      0.91      3489

    accuracy                           0.91     12513
   macro avg       0.90      0.87      0.89     12513
weighted avg       0.91      0.91      0.91     12513
```
5개 라벨 예측의 전체 정확도는 91.1%였습니다.

테스트 데이터를 사용하여 모델을 평가합니다.
모델의 예측값(y_pred)과 실제 라벨(y_test)을 비교하여 정확도를 계산합니다.
classification_report 함수를 사용하여 라벨별 정밀도, 재현율, F1 점수를 계산하고 출력합니다.

전체 정확도(Accuracy)는 91.1%로 나타났습니다.
각 라벨에 대한 세부적인 성능 지표(Precision, Recall, F1-score)도 출력되어 있습니다.
예를 들어, 'in' 라벨의 경우 정밀도(Precision)는 0.91, 재현율(Recall)은 0.87, F1-score는 0.89입니다.
전체적으로는 macro avg와 weighted avg를 통해 각 라벨의 성능을 종합적으로 확인할 수 있습니다.
 ###  **1-2. 랜덤포레스트 피쳐 가공 모델**
#### 1-2-1. R을 이용한 피쳐 평균치 모델
피쳐의 갯수를 줄이고자, Hz를 일정구간마다 나누어 평균을 내주었고, 이를 피쳐로 채택하여 정확도의 변화를 확인하고자 했습니다.
```
library('ggplot2') # visualization
library('ggthemes') # visualization
library('scales') # visualization
library('dplyr') # data manipulation
library('randomForest') # classification algorithm

#사용할 데이터 불러오기
train <- read.csv('./training.csv', stringsAsFactors = F)
test <- read.csv('./test.csv', stringsAsFactors = F)


#train파일에서 행별 구간 평균 계산 (100HZ 단위로 묶어 평균 계산)
train$mean0 <- rowMeans(train[,c('X0HZ' ,'X10HZ','X20HZ','X30HZ','X40HZ',
                                 'X50HZ','X60HZ','X70HZ','X80HZ','X90HZ')])                            
train$mean1 <- rowMeans(train[,c('X100HZ' ,'X110HZ','X120HZ','X130HZ','X140HZ',
                                 'X150HZ' ,'X160HZ','X170HZ','X180HZ','X190HZ')])


#test파일도 마찬가지로 평균계산.
test$mean0 <- rowMeans(test[,c('X0HZ' ,'X10HZ','X20HZ','X30HZ','X40HZ',
                                 'X50HZ','X60HZ','X70HZ','X80HZ','X90HZ')])                                 
test$mean1 <- rowMeans(test[,c('X100HZ' ,'X110HZ','X120HZ','X130HZ','X140HZ',
                                 'X150HZ' ,'X160HZ','X170HZ','X180HZ','X190HZ')])


#임의의 seed값 배정
set.seed(456)

#랜덤 포레스트 모델 형성
system.time(rf_model <- randomForest(factor(leaktype) ~
                           #부가 정보
                           site + sid + ldate + lrate + llevel +
                           
                           #각 구간별 HZ 평균값
                           mean0 + mean1 + mean2 + mean3 + mean4 +
                           mean5 + mean6 + mean7 + mean8 + mean9 +
                           mean10 + mean11 + mean12 + mean13 + mean14 +
                           mean15 + mean16 + mean17 + mean18 + mean19 +
                           mean20 + mean21 + mean22 + mean23 + mean24 +
                           mean25 + mean26 + mean27 + mean28 + mean29 +
                           mean30 + mean31 + mean32 + mean33 + mean34 +
                           mean35 + mean36 + mean37 + mean38 + mean39 +
                           mean40 + mean41 + mean42 + mean43 + mean44 +
                           mean45 + mean46 + mean47 + mean48 + mean49 +
                           mean50 + mean51 +

                           #MAX0 - MAX19
                           MAX0 + MAX1 + MAX2 + MAX3 + MAX4 + 
                           MAX5 + MAX6 + MAX7 + MAX8 + MAX9 + 
                           MAX10 + MAX11 + MAX12 + MAX13 + MAX14 + 
                           MAX15 + MAX16 + MAX17 + MAX18 + MAX19 ,
                          
                           data = train)
                        
#모델 에러 표시
plot(rf_model, ylim=c(0,0.36))
legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)

#중요도 분석
importance    <- importance(rf_model)
varImportance <- data.frame(Variables = row.names(importance),                            
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

#각 변수 별 중요도
rankImportance <- varImportance %>%  
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

#test파일 예측
prediction <- predict(rf_model, test)

#예측을 포함한 데이터 프레임 생성
solution <- data.frame(site = test$site,
                       sid = test$sid,
                       ldate = test$ldate,
                       lrate = test$lrate,
                       llevel = test$llevel,
                       leaktype = test$leaktype,
                       prediction = prediction,
                       accurate = ifelse(test$leaktype == prediction,0,1))


true=count(solution,accurate==0)
false=count(solution,accurate==1)
accuracy = true/(true+false)

#정답률
accuracy[2,2]

#파일 출력
write.csv(solution, file = 'leak_solution.csv', row.names = F)
theme_few()
```
![image](https://github.com/DLProject-HYU/DLProject-HYU.github.io/assets/149747730/5e52bcc0-9093-4775-a322-072507eaecad)
```
accurate == 0         n
           NaN 0.1317136
           0.5 0.8682864
```

#### 1-2-2. Python을 이용하여 중요한 피쳐를 골라낸 모델
0~5120Hz 범위의 소리를 10Hz 단위로 측정하고, Max값 또한 20개를 포함한 데이터이다보니 컬럼의 수가 너무 많았습니다. 따라서 중요한 데이터를 찾아낼 필요가 있었습니다.
```
import matplotlib.pyplot as plt
features = X.columns

plt.figure(figsize=(10, 6))
plt.bar(features, rf_classifier.feature_importances_, color='skyblue')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.xticks(features)
plt.show()
```
![image](https://github.com/DLProject-HYU/DLProject-HYU.github.io/assets/149747730/c3ec171e-97af-4861-b1a8-54f8fffd794e)


저음 영역대와 Max 값들의 중요도가 높음을 볼 수 있습니다.
따라서, 0~790Hz와 Max값들만을 사용하여 최적화를 진행했습니다.
```
X_train_2 = pd.concat([X_train.iloc[:, :80], X_train.iloc[:, -20:]], axis=1)
X_test_2 = pd.concat([X_test.iloc[:, :80], X_test.iloc[:, -20:]], axis=1)

rf_classifier_2 = RandomForestClassifier(random_state=42)
rf_classifier_2.fit(X_train_2, y_train)

y_pred = rf_classifier_2.predict(X_test_2)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```
```
Accuracy: 0.9303923919124111
```
```
print(classification_report(y_test, y_pred))
```
```
              precision    recall  f1-score   support

          in       0.92      0.91      0.91      2622
       noise       0.87      0.82      0.85       988
      normal       0.97      1.00      0.98      3933
       other       0.94      0.81      0.87      1481
         out       0.91      0.95      0.93      3489

    accuracy                           0.93     12513
   macro avg       0.92      0.90      0.91     12513
weighted avg       0.93      0.93      0.93     12513
```

전체 정확도 뿐만 아니라, 각각의 라벨들에 대한 정확도 모두 향상되었음을 확인할 수 있습니다.

이 코드와 방금 전의 코드 사이의 차이점은 다음과 같습니다:

피쳐 엔지니어링:
이 코드에서는 주어진 Hz 범위에 대해 각 행별로 평균을 계산하여 새로운 피쳐를 생성했습니다.
이것은 원래 데이터의 다양한 Hz 범위에서 추출한 피쳐를 특정 구간으로 줄이는 효과가 있습니다.

# 2.python을 이용한 Randomforest,xgboost 사용코드 
###   **2-1. XGBoost 피쳐 가공 모델**

예측 성능을 올리기 위해, Randomforest 모델 대신 XGBoost를 사용해봤습니다.
```
pip install xgboost
```
```
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test.values.ravel())

xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_classifier.fit(X_train_2, y_train_encoded)

y_pred = xgb_classifier.predict(X_test_2)
accuracy = accuracy_score(y_test_encoded, y_pred)

print("Accuracy:", accuracy)
```
```
Accuracy: 0.8955486294253976
```

오히려 정확도가 낮아졌는데, 이는 XGBoost가 많은 하이퍼파라미터를 가지고 있고, 이에 민감한 알고리즘 때문입니다.

XGBoost 모델 적용: XGBoost 라이브러리에서 XGBClassifier를 사용하여 모델을 생성합니다.
1.라벨 인코딩: LabelEncoder를 사용하여 범주형 라벨을 숫자로 변환합니다.
2.모델 학습: 훈련 데이터에 대해 XGBoost 모델을 학습시킵니다.
3.예측 및 정확도 평가: 테스트 데이터를 사용하여 모델을 평가하고 정확도를 계산합니다.
결과 해석:
정확도: 0.8955
초기 XGBoost 모델은 기본 하이퍼파라미터를 사용하며, 성능이 Random Forest보다 떨어진 것으로 보입니다.

###   **2-2. XGBoost 피쳐 가공+ 하이퍼파라미터 튜닝 모델**

XGBoost는 연산 비용이 높은 알고리즘이기 때문에, Grid-Search나 Random-Search보다 효율적인 베이지안 최적화를 이용했습니다.
```
pip install bayesian-optimization
```
베이지안 최적화를 통해, XGBoost의 하이퍼파라미터를 개선시켰습니다.
```
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score

def xgb_eval(eta, min_child_weight, gamma, max_depth, colsample_bytree, lambda_, alpha):
    params = {
        'eta': eta,
        'min_child_weight': min_child_weight,
        'gamma': gamma,
        'max_depth': int(max_depth),
        'colsample_bytree': colsample_bytree,
        'lambda': lambda_,
        'alpha': alpha,
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state': 42
    }
    xgb_model = XGBClassifier(**params)
    scores = cross_val_score(xgb_model, X_train_2, y_train_encoded, cv=3, scoring='accuracy')
    return scores.mean()

pbounds = {
    'eta': (0.001, 1.),
    'min_child_weight': (0, 10),
    'gamma': (0, 5),
    'max_depth': (3, 50),
    'colsample_bytree': (0.3, 1.0),
    'lambda_': (0, 3),
    'alpha': (0, 3),
}

optimizer = BayesianOptimization(f=xgb_eval, pbounds=pbounds, random_state=42)

optimizer.maximize(init_points=5, n_iter=200)
```
```
optimizer.max
```
```
{'target': 0.93230908104687,
 'params': {'alpha': 0.5893547220720841,
  'colsample_bytree': 1.0,
  'eta': 0.20334070944650803,
  'gamma': 0.0,
  'lambda_': 2.7303408686032196,
  'max_depth': 27.90916614548846,
  'min_child_weight': 7.301529163320516}}
```
```
params = {
    'alpha': 0.5893547220720841,
    'colsample_bytree': 1.0,
    'eta': 0.20334070944650803,
    'gamma': 0.0,
    'lambda': 2.7303408686032196,
    'max_depth': 27,
    'min_child_weight': 7.301529163320516,
    'eval_metric': 'mlogloss',
    'use_label_encoder': False,
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': 42
}

xgb_classifier = XGBClassifier(**params)
xgb_classifier.fit(X_train_2, y_train_encoded)

y_pred = xgb_classifier.predict(X_test_2)
accuracy = accuracy_score(y_test_encoded, y_pred)

print("Accuracy:", accuracy)
```
```
Accuracy: 0.9446175976983937
```
베이지안 최적화를 통해 찾은 최적 하이퍼파라미터를 적용한 XGBoost 모델은 정확도가 상당히 향상되었습니다.

2-1과 2-2의 차이:

2-1에서는 기본 XGBoost 모델을 사용하고, 2-2에서는 베이지안 최적화를 통해 하이퍼파라미터를 튜닝한 모델을 사용했습니다.
2-2에서는 더 높은 정확도를 달성했습니다. 이는 최적화된 하이퍼파라미터가 모델의 성능을 향상시켰음을 나타냅니다.
2-2에서 사용된 최적 하이퍼파라미터는 {'alpha': 0.589354..., 'colsample_bytree': 1.0, 'eta': 0.203340..., 'gamma': 0.0, 'lambda_': 2.730340..., 'max_depth': 27, 'min_child_weight': 7.301529...}입니다.
지금까지의 모델들의 정확도를 비교해보면
```
랜덤 포레스트 기본 모델 : Accuracy: 0.9114520898265803
랜덤 포레스트 피쳐 가공 : Accuracy: 0.9303923919124111
XGBoost 하이퍼파라미터 최적화 전 (피쳐 가공) : Accuracy: 0.8955486294253976
XGBoost 하이퍼파라미터 최적화 후 (피쳐 가공) : Accuracy: 0.9446175976983937
```
베이지언 최적화를 통해 하이퍼파라미터를 최적화 해주었을 경우 랜덤 포레스트 모델보다 높은 정확도를 보임을 확인했습니다.
#### 3. 성능 비교
AIHub의 상관누수 데이터에서는 별도의 Test Data를 제공하기 때문에, 랜덤 포레스트 기본모델, 랜덤포레스트 피쳐 가공 모델, XGBoost 피쳐 가공 모델, XGBoost 피쳐 가공 + 하이퍼파라미터 튜닝 모델 4가지의 성능을 비교해봤습니다.

과정 설명:
```
베이지안 최적화 함수 정의: 주어진 범위 내에서 하이퍼파라미터에 대한 목적 함수를 정의합니다.
베이지안 최적화 수행: Bayesian Optimization 라이브러리를 사용하여 XGBoost 모델의 최적 하이퍼파라미터를 찾습니다.
최적 하이퍼파라미터 적용: 찾은 최적 하이퍼파라미터로 XGBoost 모델을 생성합니다.
모델 학습: 최적화된 모델을 훈련 데이터에 대해 학습시킵니다.
예측 및 정확도 평가: 테스트 데이터를 사용하여 최종 모델을 평가하고 정확도를 계산합니다.
```
결과 해석:

#### 3-1. 랜덤 포레스트 기본 모델
```
out_test = pd.read_csv('Data/Test/1.옥외누수(out-test).csv')
in_test = pd.read_csv('Data/Test/2.옥내누수(in-test).csv')
noise_test = pd.read_csv('Data/Test/3.기계.전기음(noise-test).csv')
other_test = pd.read_csv('Data/Test/4.환경음(other-test).csv')
normal_test = pd.read_csv('Data/Test/5.정상음(normal-test).csv')

out_test.drop(['site', 'sid', 'ldate', 'lrate', 'llevel'], axis=1, inplace=True)
in_test.drop(['site', 'sid', 'ldate', 'lrate', 'llevel'], axis=1, inplace=True)
noise_test.drop(['site', 'sid', 'ldate', 'lrate', 'llevel'], axis=1, inplace=True)
other_test.drop(['site', 'sid', 'ldate', 'lrate', 'llevel'], axis=1, inplace=True)
normal_test.drop(['site', 'sid', 'ldate', 'lrate', 'llevel'], axis=1, inplace=True)

test_final = pd.concat([out_test, in_test, noise_test, other_test, normal_test]).reset_index(drop=True)
x_test_f = test_final.drop('leaktype', axis=1)

y_test_f = test_final['leaktype']
y_test_f_encoded = label_encoder.transform(y_test_f.values.ravel())
y_pred = rf_classifier.predict(x_test_f)
accuracy = accuracy_score(y_test_f, y_pred)
print("Accuracy:", accuracy)
```
```
Accuracy: 0.9043478260869565
```
#### 3-2. 랜덤 포레스트 피쳐 가공
```
x_test_f = pd.concat([x_test_f.iloc[:, :80], x_test_f.iloc[:, -20:]], axis=1)
y_pred = rf_classifier_2.predict(x_test_f)
accuracy = accuracy_score(y_test_f, y_pred)
print("Accuracy:", accuracy)
```
```
Accuracy: 0.9273657289002557
```
#### 3-3. XGBoost 피쳐 가공 모델
```
print(classification_report(y_test_f_encoded, y_pred))
```
```
              precision    recall  f1-score   support

           0       0.91      0.92      0.92      1659
           1       0.90      0.84      0.87       629
           2       1.00      1.00      1.00      2462
           3       0.93      0.85      0.89       878
           4       0.92      0.96      0.94      2192

    accuracy                           0.94      7820
   macro avg       0.93      0.91      0.92      7820
weighted avg       0.94      0.94      0.94      7820
```
```
label_mapping = {label: index for index, label in enumerate(label_encoder.classes_)}
print(label_mapping)
```
```
{'in': 0, 'noise': 1, 'normal': 2, 'other': 3, 'out': 4}
```
```
xgb_classifier_base = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_classifier_base.fit(X_train_2, y_train_encoded)

y_pred = xgb_classifier_base.predict(x_test_f)
accuracy = accuracy_score(y_test_f_encoded, y_pred)
print("Accuracy:", accuracy)
```
```
Accuracy: 0.8928388746803069
```
#### 3-4. XGBoost 피쳐 가공 및 하이퍼파라미터 최적화 모델
```
print(classification_report(y_test_f, y_pred))
```
```
              precision    recall  f1-score   support

          in       0.92      0.90      0.91      1659
       noise       0.89      0.80      0.84       629
      normal       0.97      1.00      0.98      2462
       other       0.93      0.79      0.86       878
         out       0.90      0.96      0.93      2192

    accuracy                           0.93      7820
   macro avg       0.92      0.89      0.90      7820
weighted avg       0.93      0.93      0.93      7820
```
```
y_pred = xgb_classifier.predict(x_test_f)
accuracy = accuracy_score(y_test_f_encoded, y_pred)
print("Accuracy:", accuracy)
```
```
Accuracy: 0.9425831202046036
```

결론:
베이지안 최적화를 통해 튜닝된 XGBoost 모델이 성능이 가장 우수하다고 할 수 있습니다.
최적화된 모델은 더 높은 정확도를 보여주며, 하이퍼파라미터 튜닝이 모델의 성능 향상에 기여한 것으로 판단됩니다.

# Related Work

# Conclusion: Discussion
