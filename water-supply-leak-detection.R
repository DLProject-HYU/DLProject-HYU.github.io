
library('ggplot2') # visualization
library('ggthemes') # visualization
library('scales') # visualization
library('dplyr') # data manipulation
library('randomForest') # classification algorithm

# 파일 입력
train <- read.csv('./training.csv', stringsAsFactors = F)
test <- read.csv('./test.csv', stringsAsFactors = F)

# 입력 확인
str(train)
str(test)

#train파일에서 행별 구간 평균 계산 (100HZ 단위로 묶어 평균 계산)
train$mean0 <- rowMeans(train[,c('X0HZ' ,'X10HZ','X20HZ','X30HZ','X40HZ',
                                 'X50HZ','X60HZ','X70HZ','X80HZ','X90HZ')])
train$mean1 <- rowMeans(train[,c('X100HZ' ,'X110HZ','X120HZ','X130HZ','X140HZ',
                                 'X150HZ' ,'X160HZ','X170HZ','X180HZ','X190HZ')])
train$mean2 <- rowMeans(train[,c('X200HZ' ,'X210HZ','X220HZ','X230HZ','X240HZ',
                                 'X250HZ' ,'X260HZ','X270HZ','X280HZ','X290HZ')])
train$mean3 <- rowMeans(train[,c('X300HZ' ,'X310HZ','X320HZ','X330HZ','X340HZ',
                                 'X350HZ' ,'X360HZ','X370HZ','X380HZ','X390HZ')])
train$mean4 <- rowMeans(train[,c('X400HZ' ,'X410HZ','X420HZ','X430HZ','X440HZ',
                                 'X450HZ' ,'X460HZ','X470HZ','X480HZ','X490HZ')])
train$mean5 <- rowMeans(train[,c('X500HZ' ,'X510HZ','X520HZ','X530HZ','X540HZ',
                                 'X550HZ' ,'X560HZ','X570HZ','X580HZ','X590HZ')])
train$mean6 <- rowMeans(train[,c('X600HZ' ,'X610HZ','X620HZ','X630HZ','X640HZ',
                                 'X650HZ' ,'X660HZ','X670HZ','X680HZ','X690HZ')])
train$mean7 <- rowMeans(train[,c('X700HZ' ,'X710HZ','X720HZ','X730HZ','X740HZ',
                                 'X750HZ' ,'X760HZ','X770HZ','X780HZ','X790HZ')])
train$mean8 <- rowMeans(train[,c('X800HZ' ,'X810HZ','X820HZ','X830HZ','X840HZ',
                                 'X850HZ' ,'X860HZ','X870HZ','X880HZ','X890HZ')])
train$mean9 <- rowMeans(train[,c('X900HZ' ,'X910HZ','X920HZ','X930HZ','X940HZ',
                                 'X950HZ' ,'X960HZ','X970HZ','X980HZ','X990HZ')])
train$mean10 <- rowMeans(train[,c('X1000HZ','X1010HZ','X1020HZ','X1030HZ','X1040HZ',
                                  'X1050HZ','X1060HZ','X1070HZ','X1080HZ','X1090HZ')])
train$mean11 <- rowMeans(train[,c('X1100HZ','X1110HZ','X1120HZ','X1130HZ','X1140HZ',
                                  'X1150HZ','X1160HZ','X1170HZ','X1180HZ','X1190HZ')])
train$mean12 <- rowMeans(train[,c('X1200HZ','X1210HZ','X1220HZ','X1230HZ','X1240HZ',
                                  'X1250HZ','X1260HZ','X1270HZ','X1280HZ','X1290HZ')])
train$mean13 <- rowMeans(train[,c('X1300HZ','X1310HZ','X1320HZ','X1330HZ','X1340HZ',
                                  'X1350HZ','X1360HZ','X1370HZ','X1380HZ','X1390HZ')])
train$mean14 <- rowMeans(train[,c('X1400HZ','X1410HZ','X1420HZ','X1430HZ','X1440HZ',
                                  'X1450HZ','X1460HZ','X1470HZ','X1480HZ','X1490HZ')])
train$mean15 <- rowMeans(train[,c('X1500HZ','X1510HZ','X1520HZ','X1530HZ','X1540HZ',
                                  'X1550HZ','X1560HZ','X1570HZ','X1580HZ','X1590HZ')])
train$mean16 <- rowMeans(train[,c('X1600HZ','X1610HZ','X1620HZ','X1630HZ','X1640HZ',
                                  'X1650HZ','X1660HZ','X1670HZ','X1680HZ','X1690HZ')])
train$mean17 <- rowMeans(train[,c('X1700HZ','X1710HZ','X1720HZ','X1730HZ','X1740HZ',
                                  'X1750HZ','X1760HZ','X1770HZ','X1780HZ','X1790HZ')])
train$mean18 <- rowMeans(train[,c('X1800HZ','X1810HZ','X1820HZ','X1830HZ','X1840HZ',
                                  'X1850HZ','X1860HZ','X1870HZ','X1880HZ','X1890HZ')])
train$mean19 <- rowMeans(train[,c('X1900HZ','X1910HZ','X1920HZ','X1930HZ','X1940HZ',
                                  'X1950HZ','X1960HZ','X1970HZ','X1980HZ','X1990HZ')])
train$mean20 <- rowMeans(train[,c('X2000HZ','X2010HZ','X2020HZ','X2030HZ','X2040HZ',
                                  'X2050HZ','X2060HZ','X2070HZ','X2080HZ','X2090HZ')])
train$mean21 <- rowMeans(train[,c('X2100HZ','X2110HZ','X2120HZ','X2130HZ','X2140HZ',
                                  'X2150HZ','X2160HZ','X2170HZ','X2180HZ','X2190HZ')])
train$mean22 <- rowMeans(train[,c('X2200HZ','X2210HZ','X2220HZ','X2230HZ','X2240HZ',
                                  'X2250HZ','X2260HZ','X2270HZ','X2280HZ','X2290HZ')])
train$mean23 <- rowMeans(train[,c('X2300HZ','X2310HZ','X2320HZ','X2330HZ','X2340HZ',
                                  'X2350HZ','X2360HZ','X2370HZ','X2380HZ','X2390HZ')])
train$mean24 <- rowMeans(train[,c('X2400HZ','X2410HZ','X2420HZ','X2430HZ','X2440HZ',
                                  'X2450HZ','X2460HZ','X2470HZ','X2480HZ','X2490HZ')])
train$mean25 <- rowMeans(train[,c('X2500HZ','X2510HZ','X2520HZ','X2530HZ','X2540HZ',
                                  'X2550HZ','X2560HZ','X2570HZ','X2580HZ','X2590HZ')])
train$mean26 <- rowMeans(train[,c('X2600HZ','X2610HZ','X2620HZ','X2630HZ','X2640HZ',
                                  'X2650HZ','X2660HZ','X2670HZ','X2680HZ','X2690HZ')])
train$mean27 <- rowMeans(train[,c('X2700HZ','X2710HZ','X2720HZ','X2730HZ','X2740HZ',
                                  'X2750HZ','X2760HZ','X2770HZ','X2780HZ','X2790HZ')])
train$mean28 <- rowMeans(train[,c('X2800HZ','X2810HZ','X2820HZ','X2830HZ','X2840HZ',
                                  'X2850HZ','X2860HZ','X2870HZ','X2880HZ','X2890HZ')])
train$mean29 <- rowMeans(train[,c('X2900HZ','X2910HZ','X2920HZ','X2930HZ','X2940HZ',
                                  'X2950HZ','X2960HZ','X2970HZ','X2980HZ','X2990HZ')])
train$mean30 <- rowMeans(train[,c('X3000HZ','X3010HZ','X3020HZ','X3030HZ','X3040HZ',
                                  'X3050HZ','X3060HZ','X3070HZ','X3080HZ','X3090HZ')])
train$mean31 <- rowMeans(train[,c('X3100HZ','X3110HZ','X3120HZ','X3130HZ','X3140HZ',
                                  'X3150HZ','X3160HZ','X3170HZ','X3180HZ','X3190HZ')])
train$mean32 <- rowMeans(train[,c('X3200HZ','X3210HZ','X3220HZ','X3230HZ','X3240HZ',
                                  'X3250HZ','X3260HZ','X3270HZ','X3280HZ','X3290HZ')])
train$mean33 <- rowMeans(train[,c('X3300HZ','X3310HZ','X3320HZ','X3330HZ','X3340HZ',
                                  'X3350HZ','X3360HZ','X3370HZ','X3380HZ','X3390HZ')])
train$mean34 <- rowMeans(train[,c('X3400HZ','X3410HZ','X3420HZ','X3430HZ','X3440HZ',
                                  'X3450HZ','X3460HZ','X3470HZ','X3480HZ','X3490HZ')])
train$mean35 <- rowMeans(train[,c('X3500HZ','X3510HZ','X3520HZ','X3530HZ','X3540HZ',
                                  'X3550HZ','X3560HZ','X3570HZ','X3580HZ','X3590HZ')])
train$mean36 <- rowMeans(train[,c('X3600HZ','X3610HZ','X3620HZ','X3630HZ','X3640HZ',
                                  'X3650HZ','X3660HZ','X3670HZ','X3680HZ','X3690HZ')])
train$mean37 <- rowMeans(train[,c('X3700HZ','X3710HZ','X3720HZ','X3730HZ','X3740HZ',
                                  'X3750HZ','X3760HZ','X3770HZ','X3780HZ','X3790HZ')])
train$mean38 <- rowMeans(train[,c('X3800HZ','X3810HZ','X3820HZ','X3830HZ','X3840HZ',
                                  'X3850HZ','X3860HZ','X3870HZ','X3880HZ','X3890HZ')])
train$mean39 <- rowMeans(train[,c('X3900HZ','X3910HZ','X3920HZ','X3930HZ','X3940HZ',
                                  'X3950HZ','X3960HZ','X3970HZ','X3980HZ','X3990HZ')])
train$mean40 <- rowMeans(train[,c('X4000HZ','X4010HZ','X4020HZ','X4030HZ','X4040HZ',
                                  'X4050HZ','X4060HZ','X4070HZ','X4080HZ','X4090HZ')])
train$mean41 <- rowMeans(train[,c('X4100HZ','X4110HZ','X4120HZ','X4130HZ','X4140HZ',
                                  'X4150HZ','X4160HZ','X4170HZ','X4180HZ','X4190HZ')])
train$mean42 <- rowMeans(train[,c('X4200HZ','X4210HZ','X4220HZ','X4230HZ','X4240HZ',
                                  'X4250HZ','X4260HZ','X4270HZ','X4280HZ','X4290HZ')])
train$mean43 <- rowMeans(train[,c('X4300HZ','X4310HZ','X4320HZ','X4330HZ','X4340HZ',
                                  'X4350HZ','X4360HZ','X4370HZ','X4380HZ','X4390HZ')])
train$mean44 <- rowMeans(train[,c('X4400HZ','X4410HZ','X4420HZ','X4430HZ','X4440HZ',
                                  'X4450HZ','X4460HZ','X4470HZ','X4480HZ','X4490HZ')])
train$mean45 <- rowMeans(train[,c('X4500HZ','X4510HZ','X4520HZ','X4530HZ','X4540HZ',
                                  'X4550HZ','X4560HZ','X4570HZ','X4580HZ','X4590HZ')])
train$mean46 <- rowMeans(train[,c('X4600HZ','X4610HZ','X4620HZ','X4630HZ','X4640HZ',
                                  'X4650HZ','X4660HZ','X4670HZ','X4680HZ','X4690HZ')])
train$mean47 <- rowMeans(train[,c('X4700HZ','X4710HZ','X4720HZ','X4730HZ','X4740HZ',
                                  'X4750HZ','X4760HZ','X4770HZ','X4780HZ','X4790HZ')])
train$mean48 <- rowMeans(train[,c('X4800HZ','X4810HZ','X4820HZ','X4830HZ','X4840HZ',
                                  'X4850HZ','X4860HZ','X4870HZ','X4880HZ','X4890HZ')])
train$mean49 <- rowMeans(train[,c('X4900HZ','X4910HZ','X4920HZ','X4930HZ','X4940HZ',
                                  'X4950HZ','X4960HZ','X4970HZ','X4980HZ','X4990HZ')])
train$mean50 <- rowMeans(train[,c('X5000HZ','X5010HZ','X5020HZ','X5030HZ','X5040HZ',
                                  'X5050HZ','X5060HZ','X5070HZ','X5080HZ','X5090HZ')])
train$mean51 <- rowMeans(train[,c('X5100HZ','X5110HZ','X5120HZ')])



#test파일에서도 행별 구간 평균 계산
test$mean0 <- rowMeans(test[,c('X0HZ' ,'X10HZ','X20HZ','X30HZ','X40HZ',
                                 'X50HZ','X60HZ','X70HZ','X80HZ','X90HZ')])
test$mean1 <- rowMeans(test[,c('X100HZ' ,'X110HZ','X120HZ','X130HZ','X140HZ',
                                 'X150HZ' ,'X160HZ','X170HZ','X180HZ','X190HZ')])
test$mean2 <- rowMeans(test[,c('X200HZ' ,'X210HZ','X220HZ','X230HZ','X240HZ',
                                 'X250HZ' ,'X260HZ','X270HZ','X280HZ','X290HZ')])
test$mean3 <- rowMeans(test[,c('X300HZ' ,'X310HZ','X320HZ','X330HZ','X340HZ',
                                 'X350HZ' ,'X360HZ','X370HZ','X380HZ','X390HZ')])
test$mean4 <- rowMeans(test[,c('X400HZ' ,'X410HZ','X420HZ','X430HZ','X440HZ',
                                 'X450HZ' ,'X460HZ','X470HZ','X480HZ','X490HZ')])
test$mean5 <- rowMeans(test[,c('X500HZ' ,'X510HZ','X520HZ','X530HZ','X540HZ',
                                 'X550HZ' ,'X560HZ','X570HZ','X580HZ','X590HZ')])
test$mean6 <- rowMeans(test[,c('X600HZ' ,'X610HZ','X620HZ','X630HZ','X640HZ',
                                 'X650HZ' ,'X660HZ','X670HZ','X680HZ','X690HZ')])
test$mean7 <- rowMeans(test[,c('X700HZ' ,'X710HZ','X720HZ','X730HZ','X740HZ',
                                 'X750HZ' ,'X760HZ','X770HZ','X780HZ','X790HZ')])
test$mean8 <- rowMeans(test[,c('X800HZ' ,'X810HZ','X820HZ','X830HZ','X840HZ',
                                 'X850HZ' ,'X860HZ','X870HZ','X880HZ','X890HZ')])
test$mean9 <- rowMeans(test[,c('X900HZ' ,'X910HZ','X920HZ','X930HZ','X940HZ',
                                 'X950HZ' ,'X960HZ','X970HZ','X980HZ','X990HZ')])
test$mean10 <- rowMeans(test[,c('X1000HZ','X1010HZ','X1020HZ','X1030HZ','X1040HZ',
                                  'X1050HZ','X1060HZ','X1070HZ','X1080HZ','X1090HZ')])
test$mean11 <- rowMeans(test[,c('X1100HZ','X1110HZ','X1120HZ','X1130HZ','X1140HZ',
                                  'X1150HZ','X1160HZ','X1170HZ','X1180HZ','X1190HZ')])
test$mean12 <- rowMeans(test[,c('X1200HZ','X1210HZ','X1220HZ','X1230HZ','X1240HZ',
                                  'X1250HZ','X1260HZ','X1270HZ','X1280HZ','X1290HZ')])
test$mean13 <- rowMeans(test[,c('X1300HZ','X1310HZ','X1320HZ','X1330HZ','X1340HZ',
                                  'X1350HZ','X1360HZ','X1370HZ','X1380HZ','X1390HZ')])
test$mean14 <- rowMeans(test[,c('X1400HZ','X1410HZ','X1420HZ','X1430HZ','X1440HZ',
                                  'X1450HZ','X1460HZ','X1470HZ','X1480HZ','X1490HZ')])
test$mean15 <- rowMeans(test[,c('X1500HZ','X1510HZ','X1520HZ','X1530HZ','X1540HZ',
                                  'X1550HZ','X1560HZ','X1570HZ','X1580HZ','X1590HZ')])
test$mean16 <- rowMeans(test[,c('X1600HZ','X1610HZ','X1620HZ','X1630HZ','X1640HZ',
                                  'X1650HZ','X1660HZ','X1670HZ','X1680HZ','X1690HZ')])
test$mean17 <- rowMeans(test[,c('X1700HZ','X1710HZ','X1720HZ','X1730HZ','X1740HZ',
                                  'X1750HZ','X1760HZ','X1770HZ','X1780HZ','X1790HZ')])
test$mean18 <- rowMeans(test[,c('X1800HZ','X1810HZ','X1820HZ','X1830HZ','X1840HZ',
                                  'X1850HZ','X1860HZ','X1870HZ','X1880HZ','X1890HZ')])
test$mean19 <- rowMeans(test[,c('X1900HZ','X1910HZ','X1920HZ','X1930HZ','X1940HZ',
                                  'X1950HZ','X1960HZ','X1970HZ','X1980HZ','X1990HZ')])
test$mean20 <- rowMeans(test[,c('X2000HZ','X2010HZ','X2020HZ','X2030HZ','X2040HZ',
                                  'X2050HZ','X2060HZ','X2070HZ','X2080HZ','X2090HZ')])
test$mean21 <- rowMeans(test[,c('X2100HZ','X2110HZ','X2120HZ','X2130HZ','X2140HZ',
                                  'X2150HZ','X2160HZ','X2170HZ','X2180HZ','X2190HZ')])
test$mean22 <- rowMeans(test[,c('X2200HZ','X2210HZ','X2220HZ','X2230HZ','X2240HZ',
                                  'X2250HZ','X2260HZ','X2270HZ','X2280HZ','X2290HZ')])
test$mean23 <- rowMeans(test[,c('X2300HZ','X2310HZ','X2320HZ','X2330HZ','X2340HZ',
                                  'X2350HZ','X2360HZ','X2370HZ','X2380HZ','X2390HZ')])
test$mean24 <- rowMeans(test[,c('X2400HZ','X2410HZ','X2420HZ','X2430HZ','X2440HZ',
                                  'X2450HZ','X2460HZ','X2470HZ','X2480HZ','X2490HZ')])
test$mean25 <- rowMeans(test[,c('X2500HZ','X2510HZ','X2520HZ','X2530HZ','X2540HZ',
                                  'X2550HZ','X2560HZ','X2570HZ','X2580HZ','X2590HZ')])
test$mean26 <- rowMeans(test[,c('X2600HZ','X2610HZ','X2620HZ','X2630HZ','X2640HZ',
                                  'X2650HZ','X2660HZ','X2670HZ','X2680HZ','X2690HZ')])
test$mean27 <- rowMeans(test[,c('X2700HZ','X2710HZ','X2720HZ','X2730HZ','X2740HZ',
                                  'X2750HZ','X2760HZ','X2770HZ','X2780HZ','X2790HZ')])
test$mean28 <- rowMeans(test[,c('X2800HZ','X2810HZ','X2820HZ','X2830HZ','X2840HZ',
                                  'X2850HZ','X2860HZ','X2870HZ','X2880HZ','X2890HZ')])
test$mean29 <- rowMeans(test[,c('X2900HZ','X2910HZ','X2920HZ','X2930HZ','X2940HZ',
                                  'X2950HZ','X2960HZ','X2970HZ','X2980HZ','X2990HZ')])
test$mean30 <- rowMeans(test[,c('X3000HZ','X3010HZ','X3020HZ','X3030HZ','X3040HZ',
                                  'X3050HZ','X3060HZ','X3070HZ','X3080HZ','X3090HZ')])
test$mean31 <- rowMeans(test[,c('X3100HZ','X3110HZ','X3120HZ','X3130HZ','X3140HZ',
                                  'X3150HZ','X3160HZ','X3170HZ','X3180HZ','X3190HZ')])
test$mean32 <- rowMeans(test[,c('X3200HZ','X3210HZ','X3220HZ','X3230HZ','X3240HZ',
                                  'X3250HZ','X3260HZ','X3270HZ','X3280HZ','X3290HZ')])
test$mean33 <- rowMeans(test[,c('X3300HZ','X3310HZ','X3320HZ','X3330HZ','X3340HZ',
                                  'X3350HZ','X3360HZ','X3370HZ','X3380HZ','X3390HZ')])
test$mean34 <- rowMeans(test[,c('X3400HZ','X3410HZ','X3420HZ','X3430HZ','X3440HZ',
                                  'X3450HZ','X3460HZ','X3470HZ','X3480HZ','X3490HZ')])
test$mean35 <- rowMeans(test[,c('X3500HZ','X3510HZ','X3520HZ','X3530HZ','X3540HZ',
                                  'X3550HZ','X3560HZ','X3570HZ','X3580HZ','X3590HZ')])
test$mean36 <- rowMeans(test[,c('X3600HZ','X3610HZ','X3620HZ','X3630HZ','X3640HZ',
                                  'X3650HZ','X3660HZ','X3670HZ','X3680HZ','X3690HZ')])
test$mean37 <- rowMeans(test[,c('X3700HZ','X3710HZ','X3720HZ','X3730HZ','X3740HZ',
                                  'X3750HZ','X3760HZ','X3770HZ','X3780HZ','X3790HZ')])
test$mean38 <- rowMeans(test[,c('X3800HZ','X3810HZ','X3820HZ','X3830HZ','X3840HZ',
                                  'X3850HZ','X3860HZ','X3870HZ','X3880HZ','X3890HZ')])
test$mean39 <- rowMeans(test[,c('X3900HZ','X3910HZ','X3920HZ','X3930HZ','X3940HZ',
                                  'X3950HZ','X3960HZ','X3970HZ','X3980HZ','X3990HZ')])
test$mean40 <- rowMeans(test[,c('X4000HZ','X4010HZ','X4020HZ','X4030HZ','X4040HZ',
                                  'X4050HZ','X4060HZ','X4070HZ','X4080HZ','X4090HZ')])
test$mean41 <- rowMeans(test[,c('X4100HZ','X4110HZ','X4120HZ','X4130HZ','X4140HZ',
                                  'X4150HZ','X4160HZ','X4170HZ','X4180HZ','X4190HZ')])
test$mean42 <- rowMeans(test[,c('X4200HZ','X4210HZ','X4220HZ','X4230HZ','X4240HZ',
                                  'X4250HZ','X4260HZ','X4270HZ','X4280HZ','X4290HZ')])
test$mean43 <- rowMeans(test[,c('X4300HZ','X4310HZ','X4320HZ','X4330HZ','X4340HZ',
                                  'X4350HZ','X4360HZ','X4370HZ','X4380HZ','X4390HZ')])
test$mean44 <- rowMeans(test[,c('X4400HZ','X4410HZ','X4420HZ','X4430HZ','X4440HZ',
                                  'X4450HZ','X4460HZ','X4470HZ','X4480HZ','X4490HZ')])
test$mean45 <- rowMeans(test[,c('X4500HZ','X4510HZ','X4520HZ','X4530HZ','X4540HZ',
                                  'X4550HZ','X4560HZ','X4570HZ','X4580HZ','X4590HZ')])
test$mean46 <- rowMeans(test[,c('X4600HZ','X4610HZ','X4620HZ','X4630HZ','X4640HZ',
                                  'X4650HZ','X4660HZ','X4670HZ','X4680HZ','X4690HZ')])
test$mean47 <- rowMeans(test[,c('X4700HZ','X4710HZ','X4720HZ','X4730HZ','X4740HZ',
                                  'X4750HZ','X4760HZ','X4770HZ','X4780HZ','X4790HZ')])
test$mean48 <- rowMeans(test[,c('X4800HZ','X4810HZ','X4820HZ','X4830HZ','X4840HZ',
                                  'X4850HZ','X4860HZ','X4870HZ','X4880HZ','X4890HZ')])
test$mean49 <- rowMeans(test[,c('X4900HZ','X4910HZ','X4920HZ','X4930HZ','X4940HZ',
                                  'X4950HZ','X4960HZ','X4970HZ','X4980HZ','X4990HZ')])
test$mean50 <- rowMeans(test[,c('X5000HZ','X5010HZ','X5020HZ','X5030HZ','X5040HZ',
                                  'X5050HZ','X5060HZ','X5070HZ','X5080HZ','X5090HZ')])
test$mean51 <- rowMeans(test[,c('X5100HZ','X5110HZ','X5120HZ')])


# 임의의 seed값 배정
set.seed(456)

# 랜덤 포레스트 모델 형성 (시간 측정)
# 제 컴퓨터에서는 5분 34초정도 걸렸습니다.
system.time(
rf_model <- randomForest(factor(leaktype) ~
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

)
# 모델 에러 표시
plot(rf_model, ylim=c(0,0.36))
legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)

# 중요도 분석
importance    <- importance(rf_model)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

# 각 변수 별 중요도 (우측 Environment 탭에서 확인 가능)
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

# test파일 예측
prediction <- predict(rf_model, test)

# 예측을 포함한 데이터 프레임 생성
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

# 정답률
accuracy[2,2]

# 파일 출력
write.csv(solution, file = 'leak_solution.csv', row.names = F)

theme_few()


