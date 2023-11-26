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
###   **1.1. 랜덤포레스트 기본 모델**
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

  ```                         
### 1-5 모델 에러 표시

plot(rf_model, ylim=c(0,0.36))

legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)

```
### 1-6 중요도 분석
```
importance    <- importance(rf_model)

varImportance <- data.frame(Variables = row.names(importance), 
                            
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

각 변수 별 중요도

rankImportance <- varImportance %>%
  
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

```
### 1-7 test파일 예측
```
prediction <- predict(rf_model, test)
```
예측을 포함한 데이터 프레임 생성
```
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

```
### 1-8 정답률
```
accuracy[2,2]
```

### 1-9 파일 출력
```
write.csv(solution, file = 'leak_solution.csv', row.names = F)

theme_few()
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
지금까지의 모델들의 정확도를 비교해보면
```
랜덤 포레스트 기본 모델 : Accuracy: 0.9114520898265803
랜덤 포레스트 피쳐 가공 : Accuracy: 0.9303923919124111
XGBoost 하이퍼파라미터 최적화 전 (피쳐 가공) : Accuracy: 0.8955486294253976
XGBoost 하이퍼파라미터 최적화 후 (피쳐 가공) : Accuracy: 0.9446175976983937
```
베이지언 최적화를 통해 하이퍼파라미터를 최적화 해주었을 경우 랜덤 포레스트 모델보다 높은 정확도를 보임을 확인했습니다.

AIHub의 상관누수 데이터에서는 별도의 Test Data를 제공하기 때문에, 랜덤 포레스트 기본모델, 랜덤포레스트 피쳐 가공 모델, XGBoost 피쳐 가공 모델, XGBoost 피쳐 가공 + 하이퍼파라미터 튜닝 모델 4가지의 성능을 비교해봤습니다.

# Related Work

# Conclusion: Discussion
