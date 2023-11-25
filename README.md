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
# 1. R을 이용한 Randomforest 코드
1-1. 불러올 함수

library('ggplot2') # visualization

library('ggthemes') # visualization

library('scales') # visualization

library('dplyr') # data manipulation

library('randomForest') # classification algorithm


1-2 파일입력

train <- read.csv('./training.csv', stringsAsFactors = F)

test <- read.csv('./test.csv', stringsAsFactors = F)

train파일에서 행별 구간 평균 계산 (100HZ 단위로 묶어 평균 계산)

train$mean0 <- rowMeans(train[,c('X0HZ' ,'X10HZ','X20HZ','X30HZ','X40HZ',
                                 'X50HZ','X60HZ','X70HZ','X80HZ','X90HZ')])
                                 
train$mean1 <- rowMeans(train[,c('X100HZ' ,'X110HZ','X120HZ','X130HZ','X140HZ',
                                 'X150HZ' ,'X160HZ','X170HZ','X180HZ','X190HZ')])
.
.
.

test파일도 마찬가지로 평균계산.

test$mean0 <- rowMeans(test[,c('X0HZ' ,'X10HZ','X20HZ','X30HZ','X40HZ',
                                 'X50HZ','X60HZ','X70HZ','X80HZ','X90HZ')])
                                 
test$mean1 <- rowMeans(test[,c('X100HZ' ,'X110HZ','X120HZ','X130HZ','X140HZ',
                                 'X150HZ' ,'X160HZ','X170HZ','X180HZ','X190HZ')])
.
.
.


1-3임의의 seed값 배정

set.seed(456)


1-4 랜덤 포레스트 모델 형성 (시간 측정)
 
 코드 구동시 5분 34초정도 걸렸습니다.

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

                           
1-5 모델 에러 표시

plot(rf_model, ylim=c(0,0.36))

legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)


1-6 중요도 분석

importance    <- importance(rf_model)

varImportance <- data.frame(Variables = row.names(importance), 
                            
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

각 변수 별 중요도 (우측 Environment 탭에서 확인 가능)

rankImportance <- varImportance %>%
  
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))


1-7 test파일 예측

prediction <- predict(rf_model, test)

예측을 포함한 데이터 프레임 생성

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


1-8 정답률

accuracy[2,2]


1-9 파일 출력

write.csv(solution, file = 'leak_solution.csv', row.names = F)

theme_few()

# 2.python을 이용한 Randomforest,xgboost 사용코드 
# Related Work

# Conclusion: Discussion
