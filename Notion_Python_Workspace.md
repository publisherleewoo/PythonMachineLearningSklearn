# 🐍 Python Workspace - Jupyter 노트북 완전 정리

📅 **정리 날짜**: 2025년 09월 21일 21시 30분
📁 **소스 경로**: `C:\pythonworkspace`
📊 **총 노트북 수**: 6개

---

## 📑 목차

1. [01. Linear Regression.ipynb](#01.-linear-regression)
2. [02. Multiple Linear Regression.ipynb](#02.-multiple-linear-regression)
3. [03. Polynomial Regression.ipynb](#03.-polynomial-regression)
4. [04. Logistic Regression.ipynb](#04.-logistic-regression)
5. [05. K-Means.ipynb](#05.-k-means)
6. [06. Quiz.ipynb](#06.-quiz)

---

## 📓 01. Linear Regression.ipynb

> **파일 경로**: `C:\pythonworkspace\01. Linear Regression.ipynb`
> **총 셀 개수**: 43개

### 📝 셀 1 - 마크다운

# 1. Linear Regression
### 공부 시간에 따른 시험 점수

---

### 💻 셀 2 - 코드

```python
import matplotlib.pyplot as plt
import pandas as pd
```

---

### 💻 셀 3 - 코드

```python
dataset = pd.read_csv('LinearRegressionData.csv')
```

---

### 💻 셀 4 - 코드

```python
dataset.head()
```

**📤 실행 결과:**

```
hour  score
0   0.5     10
1   1.2      8
2   1.8     14
3   2.4     26
4   2.6     22
```

---

### 💻 셀 5 - 코드

```python
X = dataset.iloc[:, :-1].values # 처음부터 마지막 컬럼 직전까지의 데이터 (독립 변수 - 원인)
y = dataset.iloc[:, -1].values # 마지막 컬럼 데이터 (종속 변수 - 결과)
```

---

### 💻 셀 6 - 코드

```python
X, y
```

**📤 실행 결과:**

```
(array([[ 0.5],
        [ 1.2],
        [ 1.8],
        [ 2.4],
        [ 2.6],
        [ 3.2],
        [ 3.9],
        [ 4.4],
        [ 4.5],
        [ 5. ],
        [ 5.3],
        [ 5.8],
        [ 6. ],
        [ 6.1],
        [ 6.2],
        [ 6.9],
        [ 7.2],
        [ 8.4],
        [ 8.6],
        [10. ]]),
 array([ 10,   8,  14,  26,  22,  30,  42,  48,  38,  58,  60,  72,  62,
         68,  72,  58,  76,  86,  90, 100], dtype=int64))
```

---

### 💻 셀 7 - 코드

```python
from sklearn.linear_model import LinearRegression
reg = LinearRegression() # 객체 생성
reg.fit(X, y) # 학습 (모델 생성)
```

**📤 실행 결과:**

```
LinearRegression()
```

---

### 💻 셀 8 - 코드

```python
y_pred = reg.predict(X) # X 에 대한 예측 값
y_pred
```

**📤 실행 결과:**

```
array([  5.00336377,  12.31395163,  18.58016979,  24.84638795,
        26.93512734,  33.20134551,  40.51193337,  45.73378184,
        46.77815153,  52.        ,  55.13310908,  60.35495755,
        62.44369694,  63.48806663,  64.53243633,  71.84302419,
        74.97613327,  87.5085696 ,  89.59730899, 104.2184847 ])
```

---

### 💻 셀 9 - 코드

```python
plt.scatter(X, y, color='blue') # 산점도
plt.plot(X, y_pred, color='green') # 선 그래프
plt.title('Score by hours') # 제목
plt.xlabel('hours') # X 축 이름
plt.ylabel('score') # Y 축 이름
plt.show()
```

**📤 실행 결과:**

```
<Figure size 432x288 with 1 Axes>
```

---

### 💻 셀 10 - 코드

```python
print('9시간 공부했을 때 예상 점수 : ', reg.predict([[9]])) # [[9], [8], [7]]
```

**📤 실행 결과:**

```
9시간 공부했을 때 예상 점수 :  [93.77478776]
```

---

### 💻 셀 11 - 코드

```python
reg.coef_ # 기울기 (m)
```

**📤 실행 결과:**

```
array([10.44369694])
```

---

### 💻 셀 12 - 코드

```python
reg.intercept_ # y 절편 (b)
```

**📤 실행 결과:**

```
-0.21848470286721522
```

---

### 📝 셀 13 - 마크다운

y = mx + b  -> y = 10.4436x - 0.2184

---

### 📝 셀 14 - 마크다운

### 데이터 세트 분리

---

### 💻 셀 15 - 코드

```python
import matplotlib.pyplot as plt
import pandas as pd
```

---

### 💻 셀 16 - 코드

```python
dataset = pd.read_csv('LinearRegressionData.csv')
dataset
```

**📤 실행 결과:**

```
hour  score
0    0.5     10
1    1.2      8
2    1.8     14
3    2.4     26
4    2.6     22
5    3.2     30
6    3.9     42
7    4.4     48
8    4.5     38
9    5.0     58
10   5.3     60
11   5.8     72
12   6.0     62
13   6.1     68
14   6.2     72
15   6.9     58
16   7.2     76
17   8.4     86
18   8.6     90
19  10.0    100
```

---

### 💻 셀 17 - 코드

```python
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

---

### 💻 셀 18 - 코드

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) # 훈련 80 : 테스트 20 으로 분리
```

---

### 💻 셀 19 - 코드

```python
X, len(X) # 전체 데이터 X, 개수
```

**📤 실행 결과:**

```
(array([[ 0.5],
        [ 1.2],
        [ 1.8],
        [ 2.4],
        [ 2.6],
        [ 3.2],
        [ 3.9],
        [ 4.4],
        [ 4.5],
        [ 5. ],
        [ 5.3],
        [ 5.8],
        [ 6. ],
        [ 6.1],
        [ 6.2],
        [ 6.9],
        [ 7.2],
        [ 8.4],
        [ 8.6],
        [10. ]]),
 20)
```

---

### 💻 셀 20 - 코드

```python
X_train, len(X_train) # 훈련 세트 X, 개수
```

**📤 실행 결과:**

```
(array([[5.3],
        [8.4],
        [3.9],
        [6.1],
        [2.6],
        [1.8],
        [3.2],
        [6.2],
        [5. ],
        [4.4],
        [7.2],
        [5.8],
        [2.4],
        [0.5],
        [6.9],
        [6. ]]),
 16)
```

---

### 💻 셀 21 - 코드

```python
X_test, len(X_test) # 테스트 세트 X, 개수
```

**📤 실행 결과:**

```
(array([[ 8.6],
        [ 1.2],
        [10. ],
        [ 4.5]]),
 4)
```

---

### 💻 셀 22 - 코드

```python
y, len(y) # 전체 데이터 y
```

**📤 실행 결과:**

```
(array([ 10,   8,  14,  26,  22,  30,  42,  48,  38,  58,  60,  72,  62,
         68,  72,  58,  76,  86,  90, 100], dtype=int64),
 20)
```

---

### 💻 셀 23 - 코드

```python
y_train, len(y_train) # 훈련 세트 y
```

**📤 실행 결과:**

```
(array([60, 86, 42, 68, 22, 14, 30, 72, 58, 48, 76, 72, 26, 10, 58, 62],
       dtype=int64),
 16)
```

---

### 💻 셀 24 - 코드

```python
y_test, len(y_test) # 테스트 세트 y
```

**📤 실행 결과:**

```
(array([ 90,   8, 100,  38], dtype=int64), 4)
```

---

### 📝 셀 25 - 마크다운

### 분리된 데이터를 통한 모델링

---

### 💻 셀 26 - 코드

```python
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
```

---

### 💻 셀 27 - 코드

```python
reg.fit(X_train, y_train) # 훈련 세트로 학습
```

**📤 실행 결과:**

```
LinearRegression()
```

---

### 📝 셀 28 - 마크다운

### 데이터 시각화 (훈련 세트)

---

### 💻 셀 29 - 코드

```python
plt.scatter(X_train, y_train, color='blue') # 산점도
plt.plot(X_train, reg.predict(X_train), color='green') # 선 그래프
plt.title('Score by hours (train data)') # 제목
plt.xlabel('hours') # X 축 이름
plt.ylabel('score') # Y 축 이름
plt.show()
```

**📤 실행 결과:**

```
<Figure size 432x288 with 1 Axes>
```

---

### 📝 셀 30 - 마크다운

### 데이터 시각화 (테스트 세트)

---

### 💻 셀 31 - 코드

```python
plt.scatter(X_test, y_test, color='blue') # 산점도
plt.plot(X_train, reg.predict(X_train), color='green') # 선 그래프
plt.title('Score by hours (test data)') # 제목
plt.xlabel('hours') # X 축 이름
plt.ylabel('score') # Y 축 이름
plt.show()
```

**📤 실행 결과:**

```
<Figure size 432x288 with 1 Axes>
```

---

### 💻 셀 32 - 코드

```python
reg.coef_
```

**📤 실행 결과:**

```
array([10.49161294])
```

---

### 💻 셀 33 - 코드

```python
reg.intercept_
```

**📤 실행 결과:**

```
0.6115562905169796
```

---

### 📝 셀 34 - 마크다운

### 모델 평가

---

### 💻 셀 35 - 코드

```python
reg.score(X_test, y_test) # 테스트 세트를 통한 모델 평가
```

**📤 실행 결과:**

```
0.9727616474310156
```

---

### 💻 셀 36 - 코드

```python
reg.score(X_train, y_train) # 훈련 세트를 통한 모델 평가
```

**📤 실행 결과:**

```
0.9356663661221668
```

---

### 📝 셀 37 - 마크다운

## 경사 하강법 (Gradient Descent)

---

### 📝 셀 38 - 마크다운

max_iter : 훈련 세트 반복 횟수 (Epoch 횟수)

eta0 : 학습률 (learning rate)

---

### 💻 셀 39 - 코드

```python
from sklearn.linear_model import SGDRegressor # SGD : Stochastic Gradient Descent 확률적 경사 하강법

# 지수표기법
# 1e-3 : 0.001 (10^-3)
# 1e-4 : 0.0001 (10^-4)
# 1e+3 : 1000 (10^3)
# 1e+4 : 10000 (10^4)

# sr = SGDRegressor(max_iter=200, eta0=1e-4, random_state=0, verbose=1)
sr = SGDRegressor()
sr.fit(X_train, y_train)
```

**📤 실행 결과:**

```
SGDRegressor()
```

---

### 💻 셀 40 - 코드

```python
plt.scatter(X_train, y_train, color='blue') # 산점도
plt.plot(X_train, sr.predict(X_train), color='green') # 선 그래프
plt.title('Score by hours (train data, SGD)') # 제목
plt.xlabel('hours') # X 축 이름
plt.ylabel('score') # Y 축 이름
plt.show()
```

**📤 실행 결과:**

```
<Figure size 432x288 with 1 Axes>
```

---

### 💻 셀 41 - 코드

```python
sr.coef_, sr.intercept_
# 주의 : SGDRegressor() 객체를 생성할 때 random_state 값을 지정하지 않았으므로 결과가 다르게 나타날 수 있습니다
```

**📤 실행 결과:**

```
(array([10.2062811]), array([1.95017289]))
```

---

### 💻 셀 42 - 코드

```python
sr.score(X_test, y_test) # 테스트 세트를 통한 모델 평가 
```

**📤 실행 결과:**

```
0.9732274354250781
```

---

### 💻 셀 43 - 코드

```python
sr.score(X_train, y_train) # 훈련 세트를 통한 모델 평가 
```

**📤 실행 결과:**

```
0.9349740699430755
```

---


## 📓 02. Multiple Linear Regression.ipynb

> **파일 경로**: `C:\pythonworkspace\02. Multiple Linear Regression.ipynb`
> **총 셀 개수**: 24개

### 📝 셀 1 - 마크다운

# 2. Multiple Linear Regression

---

### 📝 셀 2 - 마크다운

### 원-핫 인코딩

---

### 💻 셀 3 - 코드

```python
import pandas as pd
```

---

### 💻 셀 4 - 코드

```python
dataset = pd.read_csv('MultipleLinearRegressionData.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

---

### 💻 셀 5 - 코드

```python
X
```

**📤 실행 결과:**

```
array([[0.5, 3, 'Home'],
       [1.2, 4, 'Library'],
       [1.8, 2, 'Cafe'],
       [2.4, 0, 'Cafe'],
       [2.6, 2, 'Home'],
       [3.2, 0, 'Home'],
       [3.9, 0, 'Library'],
       [4.4, 0, 'Library'],
       [4.5, 5, 'Home'],
       [5.0, 1, 'Cafe'],
       [5.3, 2, 'Cafe'],
       [5.8, 0, 'Cafe'],
       [6.0, 3, 'Library'],
       [6.1, 1, 'Cafe'],
       [6.2, 1, 'Library'],
       [6.9, 4, 'Home'],
       [7.2, 2, 'Cafe'],
       [8.4, 1, 'Home'],
       [8.6, 1, 'Library'],
       [10.0, 0, 'Library']], dtype=object)
```

---

### 💻 셀 6 - 코드

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [2])], remainder='passthrough')
X = ct.fit_transform(X)
X

# 1 0 : Home
# 0 1 : Library
# 0 0 : Cafe
```

**📤 실행 결과:**

```
array([[1.0, 0.0, 0.5, 3],
       [0.0, 1.0, 1.2, 4],
       [0.0, 0.0, 1.8, 2],
       [0.0, 0.0, 2.4, 0],
       [1.0, 0.0, 2.6, 2],
       [1.0, 0.0, 3.2, 0],
       [0.0, 1.0, 3.9, 0],
       [0.0, 1.0, 4.4, 0],
       [1.0, 0.0, 4.5, 5],
       [0.0, 0.0, 5.0, 1],
       [0.0, 0.0, 5.3, 2],
       [0.0, 0.0, 5.8, 0],
       [0.0, 1.0, 6.0, 3],
       [0.0, 0.0, 6.1, 1],
       [0.0, 1.0, 6.2, 1],
       [1.0, 0.0, 6.9, 4],
       [0.0, 0.0, 7.2, 2],
       [1.0, 0.0, 8.4, 1],
       [0.0, 1.0, 8.6, 1],
       [0.0, 1.0, 10.0, 0]], dtype=object)
```

---

### 📝 셀 7 - 마크다운

### 데이터 세트 분리

---

### 💻 셀 8 - 코드

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

---

### 📝 셀 9 - 마크다운

### 학습 (다중 선형 회귀)

---

### 💻 셀 10 - 코드

```python
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
```

**📤 실행 결과:**

```
LinearRegression()
```

---

### 📝 셀 11 - 마크다운

### 예측 값과 실제 값 비교 (테스트 세트)

---

### 💻 셀 12 - 코드

```python
y_pred = reg.predict(X_test)
y_pred
```

**📤 실행 결과:**

```
array([ 92.15457859,  10.23753043, 108.36245302,  38.14675204])
```

---

### 💻 셀 13 - 코드

```python
y_test
```

**📤 실행 결과:**

```
array([ 90,   8, 100,  38], dtype=int64)
```

---

### 💻 셀 14 - 코드

```python
reg.coef_
```

**📤 실행 결과:**

```
array([-5.82712824, -1.04450647, 10.40419528, -1.64200104])
```

---

### 💻 셀 15 - 코드

```python
reg.intercept_
```

**📤 실행 결과:**

```
5.365006706544747
```

---

### 📝 셀 16 - 마크다운

### 모델 평가

---

### 💻 셀 17 - 코드

```python
reg.score(X_train, y_train) # 훈련 세트
```

**📤 실행 결과:**

```
0.9623352565265527
```

---

### 💻 셀 18 - 코드

```python
reg.score(X_test, y_test) # 테스트 세트
```

**📤 실행 결과:**

```
0.9859956178877445
```

---

### 📝 셀 19 - 마크다운

### 다양한 평가 지표 (회귀 모델)

---

### 📝 셀 20 - 마크다운

1. MAE (Mean Absolute Error) : (실제 값과 예측 값) 차이의 절대값
1. MSE (Mean Squared Error) : 차이의 제곱
1. RMSE (Root Mean Squared Error) : 차이의 제곱에 루트
1. R2 : 결정 계수

> R2 는 1에 가까울수록, 나머지는 0에 가까울수록 좋음

---

### 💻 셀 21 - 코드

```python
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred) # 실제 값, 예측 값 # MAE
```

**📤 실행 결과:**

```
3.2253285188288023
```

---

### 💻 셀 22 - 코드

```python
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred) # MSE
```

**📤 실행 결과:**

```
19.900226981515015
```

---

### 💻 셀 23 - 코드

```python
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred, squared=False) # RMSE
```

**📤 실행 결과:**

```
4.460967045553578
```

---

### 💻 셀 24 - 코드

```python
from sklearn.metrics import r2_score
r2_score(y_test, y_pred) # R2
```

**📤 실행 결과:**

```
0.9859956178877445
```

---


## 📓 03. Polynomial Regression.ipynb

> **파일 경로**: `C:\pythonworkspace\03. Polynomial Regression.ipynb`
> **총 셀 개수**: 27개

### 📝 셀 1 - 마크다운

# 3. Polynomial Regression

---

### 📝 셀 2 - 마크다운

### 공부 시간에 따른 시험 점수 (우등생)

---

### 💻 셀 3 - 코드

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

---

### 💻 셀 4 - 코드

```python
dataset = pd.read_csv('PolynomialRegressionData.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

---

### 📝 셀 5 - 마크다운

## 3-1. 단순 선형 회귀 (Simple Linear Regression)

---

### 💻 셀 6 - 코드

```python
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X, y) # 전체 데이터로 학습
```

**📤 실행 결과:**

```
LinearRegression()
```

---

### 📝 셀 7 - 마크다운

### 데이터 시각화 (전체)

---

### 💻 셀 8 - 코드

```python
plt.scatter(X, y, color='blue') # 산점도
plt.plot(X, reg.predict(X), color='green') # 선 그래프
plt.title('Score by hours (genius)') # 제목
plt.xlabel('hours') # X 축 이름
plt.ylabel('score') # Y 축 이름
plt.show()
```

**📤 실행 결과:**

```
<Figure size 432x288 with 1 Axes>
```

---

### 💻 셀 9 - 코드

```python
reg.score(X, y) # 전체 데이터를 통한 모델 평가
```

**📤 실행 결과:**

```
0.8169296513411765
```

---

### 📝 셀 10 - 마크다운

## 3-2. 다항 회귀 (Polynomial Regression)

---

### 💻 셀 11 - 코드

```python
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4) # 2차
X_poly = poly_reg.fit_transform(X)
X_poly[:5] # [x] -> [x^0, x^1, x^2] -> x 가 3이라면 [1, 3, 9] 으로 변환
```

**📤 실행 결과:**

```
array([[1.0000e+00, 2.0000e-01, 4.0000e-02, 8.0000e-03, 1.6000e-03],
       [1.0000e+00, 5.0000e-01, 2.5000e-01, 1.2500e-01, 6.2500e-02],
       [1.0000e+00, 8.0000e-01, 6.4000e-01, 5.1200e-01, 4.0960e-01],
       [1.0000e+00, 9.0000e-01, 8.1000e-01, 7.2900e-01, 6.5610e-01],
       [1.0000e+00, 1.2000e+00, 1.4400e+00, 1.7280e+00, 2.0736e+00]])
```

---

### 💻 셀 12 - 코드

```python
X[:5]
```

**📤 실행 결과:**

```
array([[0.2],
       [0.5],
       [0.8],
       [0.9],
       [1.2]])
```

---

### 💻 셀 13 - 코드

```python
poly_reg.get_feature_names_out()
```

**📤 실행 결과:**

```
array(['1', 'x0', 'x0^2', 'x0^3', 'x0^4'], dtype=object)
```

---

### 💻 셀 14 - 코드

```python
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y) # 변환된 X 와 y 를 가지고 모델 생성 (학습)
```

**📤 실행 결과:**

```
LinearRegression()
```

---

### 📝 셀 15 - 마크다운

### 데이터 시각화 (변환된 X 와 y)

---

### 💻 셀 16 - 코드

```python
plt.scatter(X, y, color='blue')
plt.plot(X, lin_reg.predict(poly_reg.fit_transform(X)), color='green')
plt.title('Score by hours (genius)') # 제목
plt.xlabel('hours') # X 축 이름
plt.ylabel('score') # Y 축 이름
plt.show()
```

**📤 실행 결과:**

```
<Figure size 432x288 with 1 Axes>
```

---

### 💻 셀 17 - 코드

```python
X_range = np.arange(min(X), max(X), 0.1) # X 의 최소값에서 최대값까지의 범위를 0.1 단위로 잘라서 데이터를 생성
X_range
```

**📤 실행 결과:**

```
array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. , 1.1, 1.2, 1.3, 1.4,
       1.5, 1.6, 1.7, 1.8, 1.9, 2. , 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
       2.8, 2.9, 3. , 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4. ,
       4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7])
```

---

### 💻 셀 18 - 코드

```python
X_range.shape
```

**📤 실행 결과:**

```
(46,)
```

---

### 💻 셀 19 - 코드

```python
X[:5]
```

**📤 실행 결과:**

```
array([[0.2],
       [0.5],
       [0.8],
       [0.9],
       [1.2]])
```

---

### 💻 셀 20 - 코드

```python
X.shape
```

**📤 실행 결과:**

```
(20, 1)
```

---

### 💻 셀 21 - 코드

```python
X_range = X_range.reshape(-1, 1) # row 개수는 자동으로 계산, column 개수는 1개
X_range.shape
```

**📤 실행 결과:**

```
(46, 1)
```

---

### 💻 셀 22 - 코드

```python
X_range[:5]
```

**📤 실행 결과:**

```
array([[0.2],
       [0.3],
       [0.4],
       [0.5],
       [0.6]])
```

---

### 💻 셀 23 - 코드

```python
plt.scatter(X, y, color='blue')
plt.plot(X_range, lin_reg.predict(poly_reg.fit_transform(X_range)), color='green')
plt.title('Score by hours (genius)') # 제목
plt.xlabel('hours') # X 축 이름
plt.ylabel('score') # Y 축 이름
plt.show()
```

**📤 실행 결과:**

```
<Figure size 432x288 with 1 Axes>
```

---

### 📝 셀 24 - 마크다운

### 공부 시간에 따른 시험 성적 예측

---

### 💻 셀 25 - 코드

```python
reg.predict([[2]]) # 2시간을 공부했을 때 선형 회귀 모델의 예측
```

**📤 실행 결과:**

```
array([19.85348988])
```

---

### 💻 셀 26 - 코드

```python
lin_reg.predict(poly_reg.fit_transform([[2]])) # 2시간을 공부했을 때 다항 회귀 모델의 예측
```

**📤 실행 결과:**

```
array([8.70559135])
```

---

### 💻 셀 27 - 코드

```python
lin_reg.score(X_poly, y)
```

**📤 실행 결과:**

```
0.9782775579000045
```

---


## 📓 04. Logistic Regression.ipynb

> **파일 경로**: `C:\pythonworkspace\04. Logistic Regression.ipynb`
> **총 셀 개수**: 31개

### 📝 셀 1 - 마크다운

# 4. Logistic Regression

---

### 📝 셀 2 - 마크다운

### 공부 시간에 따른 자격증 시험 합격 가능성

---

### 💻 셀 3 - 코드

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

---

### 💻 셀 4 - 코드

```python
dataset = pd.read_csv('LogisticRegressionData.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

---

### 📝 셀 5 - 마크다운

### 데이터 분리

---

### 💻 셀 6 - 코드

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

---

### 📝 셀 7 - 마크다운

### 학습 (로지스틱 회귀 모델)

---

### 💻 셀 8 - 코드

```python
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
```

**📤 실행 결과:**

```
LogisticRegression()
```

---

### 📝 셀 9 - 마크다운

### 6시간 공부했을 때 예측?

---

### 💻 셀 10 - 코드

```python
classifier.predict([[6]])
# 결과 1 : 합격할 것으로 예측
```

**📤 실행 결과:**

```
array([1], dtype=int64)
```

---

### 💻 셀 11 - 코드

```python
classifier.predict_proba([[6]]) # 합격할 확률 출력
# 불합격 확률 14%, 합격 확률 86%
```

**📤 실행 결과:**

```
array([[0.14150735, 0.85849265]])
```

---

### 📝 셀 12 - 마크다운

### 4시간 공부했을 때 예측?

---

### 💻 셀 13 - 코드

```python
classifier.predict([[4]])
# 결과 0 : 불합격할 것으로 예측
```

**📤 실행 결과:**

```
array([0], dtype=int64)
```

---

### 💻 셀 14 - 코드

```python
classifier.predict_proba([[4]]) # 합격할 확률 출력
# 불합격 확률 62%, 합격 확률 38%
```

**📤 실행 결과:**

```
array([[0.6249966, 0.3750034]])
```

---

### 📝 셀 15 - 마크다운

### 분류 결과 예측 (테스트 세트)

---

### 💻 셀 16 - 코드

```python
y_pred = classifier.predict(X_test)
y_pred # 예측 값
```

**📤 실행 결과:**

```
array([1, 0, 1, 1], dtype=int64)
```

---

### 💻 셀 17 - 코드

```python
y_test # 실제 값 (테스트 세트)
```

**📤 실행 결과:**

```
array([1, 0, 1, 0], dtype=int64)
```

---

### 💻 셀 18 - 코드

```python
X_test # 공부 시간 (테스트 세트)
```

**📤 실행 결과:**

```
array([[ 8.6],
       [ 1.2],
       [10. ],
       [ 4.5]])
```

---

### 💻 셀 19 - 코드

```python
classifier.score(X_test, y_test) # 모델 평가
# 전체 테스트 세트 4개 중에서 분류 예측을 올바로 맞힌 개수 3개 -> 3/4 = 0.75
```

**📤 실행 결과:**

```
0.75
```

---

### 📝 셀 20 - 마크다운

### 데이터 시각화 (훈련 세트)

---

### 💻 셀 21 - 코드

```python
X_range = np.arange(min(X), max(X), 0.1) # X 의 최소값에서 최대값까지를 0.1 단위로 잘라서 데이터 생성
X_range
```

**📤 실행 결과:**

```
array([0.5, 0.6, 0.7, 0.8, 0.9, 1. , 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
       1.8, 1.9, 2. , 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3. ,
       3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4. , 4.1, 4.2, 4.3,
       4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5. , 5.1, 5.2, 5.3, 5.4, 5.5, 5.6,
       5.7, 5.8, 5.9, 6. , 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9,
       7. , 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8. , 8.1, 8.2,
       8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9. , 9.1, 9.2, 9.3, 9.4, 9.5,
       9.6, 9.7, 9.8, 9.9])
```

---

### 💻 셀 22 - 코드

```python
p = 1 / (1 + np.exp(-(classifier.coef_ * X_range + classifier.intercept_))) # y = mx + b
p
```

**📤 실행 결과:**

```
array([[0.01035705, 0.01161247, 0.01301807, 0.0145913 , 0.01635149,
        0.01832008, 0.02052073, 0.02297953, 0.02572521, 0.02878929,
        0.03220626, 0.03601375, 0.04025264, 0.04496719, 0.05020505,
        0.05601722, 0.06245802, 0.06958479, 0.07745757, 0.08613861,
        0.09569165, 0.10618106, 0.11767067, 0.13022241, 0.14389468,
        0.15874043, 0.17480509, 0.19212422, 0.2107211 , 0.23060425,
        0.25176509, 0.27417574, 0.29778732, 0.32252874, 0.34830616,
        0.3750034 , 0.40248315, 0.43058927, 0.45914989, 0.48798142,
        0.51689314, 0.54569221, 0.57418876, 0.60220088, 0.6295591 ,
        0.65611024, 0.68172044, 0.70627722, 0.72969059, 0.75189324,
        0.77283994, 0.79250621, 0.81088652, 0.82799203, 0.84384828,
        0.85849265, 0.871972  , 0.88434036, 0.89565683, 0.90598377,
        0.91538521, 0.92392546, 0.93166808, 0.93867499, 0.9450058 ,
        0.95071738, 0.95586346, 0.96049453, 0.96465764, 0.96839647,
        0.97175136, 0.97475939, 0.97745455, 0.97986786, 0.9820276 ,
        0.98395944, 0.98568665, 0.9872303 , 0.98860939, 0.98984107,
        0.9909408 , 0.99192244, 0.99279849, 0.99358014, 0.99427745,
        0.9948994 , 0.99545406, 0.99594865, 0.99638963, 0.99678276,
        0.99713321, 0.99744558, 0.997724  , 0.99797213, 0.99819325]])
```

---

### 💻 셀 23 - 코드

```python
p.shape
```

**📤 실행 결과:**

```
(1, 95)
```

---

### 💻 셀 24 - 코드

```python
X_range.shape
```

**📤 실행 결과:**

```
(95,)
```

---

### 💻 셀 25 - 코드

```python
p = p.reshape(-1) # 1차원 배열 형태로 변경
p.shape
```

**📤 실행 결과:**

```
(95,)
```

---

### 💻 셀 26 - 코드

```python
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_range, p, color='green')
plt.plot(X_range, np.full(len(X_range), 0.5), color='red') # X_range 개수만큼 0.5 로 가득찬 배열 만들기
plt.title('Probability by hours')
plt.xlabel('hours')
plt.ylabel('P')
plt.show()
```

**📤 실행 결과:**

```
<Figure size 432x288 with 1 Axes>
```

---

### 📝 셀 27 - 마크다운

### 데이터 시각화 (테스트 세트)

---

### 💻 셀 28 - 코드

```python
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_range, p, color='green')
plt.plot(X_range, np.full(len(X_range), 0.5), color='red') # X_range 개수만큼 0.5 로 가득찬 배열 만들기
plt.title('Probability by hours (test)')
plt.xlabel('hours')
plt.ylabel('P')
plt.show()
```

**📤 실행 결과:**

```
<Figure size 432x288 with 1 Axes>
```

---

### 💻 셀 29 - 코드

```python
classifier.predict_proba([[4.5]]) # 4.5 시간 공부했을 때 확률 (모델에서는 51% 확률로 합격 예측, 실제로는 불합격)
```

**📤 실행 결과:**

```
array([[0.48310686, 0.51689314]])
```

---

### 📝 셀 30 - 마크다운

### 혼동 행렬 (Confusion Matrix)

---

### 💻 셀 31 - 코드

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

# TRUE NEGATIVE (TN)       FALSE POSITIVE (FP)
# 불합격일거야 (예측)      합격일거야 (예측)
# 불합격 (실제)             불합격 (실제)

# FALSE NEGATIVE (FN)      TRUE POSITIVE (TP)
# 불합격일거야 (예측)      합격일거야 (예측)
# 합격 (실제)               합격 (실제)
```

**📤 실행 결과:**

```
array([[1, 1],
       [0, 2]], dtype=int64)
```

---


## 📓 05. K-Means.ipynb

> **파일 경로**: `C:\pythonworkspace\05. K-Means.ipynb`
> **총 셀 개수**: 26개

### 📝 셀 1 - 마크다운

# 5. K-Means

---

### 💻 셀 2 - 코드

```python
import os # 경고 대응
os.environ['OMP_NUM_THREADS'] = '1'
```

---

### 💻 셀 3 - 코드

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

---

### 💻 셀 4 - 코드

```python
dataset = pd.read_csv('KMeansData.csv')
dataset[:5]
```

**📤 실행 결과:**

```
hour  score
0  7.33     73
1  3.71     55
2  3.43     55
3  3.06     89
4  3.33     79
```

---

### 💻 셀 5 - 코드

```python
X = dataset.iloc[:, :].values
# X = dataset.values
# X = dataset.to_numpy() # 공식 홈페이지 권장
X[:5]
```

**📤 실행 결과:**

```
array([[ 7.33, 73.  ],
       [ 3.71, 55.  ],
       [ 3.43, 55.  ],
       [ 3.06, 89.  ],
       [ 3.33, 79.  ]])
```

---

### 📝 셀 6 - 마크다운

### 데이터 시각화 (전체 데이터 분포 확인)

---

### 💻 셀 7 - 코드

```python
plt.scatter(X[:, 0], X[:, 1]) # x축 : hour, y축 : score
plt.title('Score by hours')
plt.xlabel('hours')
plt.ylabel('score')
plt.show()
```

**📤 실행 결과:**

```
<Figure size 432x288 with 1 Axes>
```

---

### 📝 셀 8 - 마크다운

### 데이터 시각화 (축 범위 통일)

---

### 💻 셀 9 - 코드

```python
plt.scatter(X[:, 0], X[:, 1]) # x축 : hour, y축 : score
plt.title('Score by hours')
plt.xlabel('hours')
plt.xlim(0, 100)
plt.ylabel('score')
plt.ylim(0, 100)
plt.show()
```

**📤 실행 결과:**

```
<Figure size 432x288 with 1 Axes>
```

---

### 📝 셀 10 - 마크다운

### 피처 스케일링 (Feature Scaling)

---

### 💻 셀 11 - 코드

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X[:5]
```

**📤 실행 결과:**

```
array([[ 0.68729921,  0.73538376],
       [-0.66687438,  0.04198891],
       [-0.77161709,  0.04198891],
       [-0.9100271 ,  1.35173473],
       [-0.8090252 ,  0.96651537]])
```

---

### 📝 셀 12 - 마크다운

### 데이터 시각화 (스케일링된 데이터)

---

### 💻 셀 13 - 코드

```python
plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1])
plt.title('Score by hours')
plt.xlabel('hours')
plt.ylabel('score')
plt.show()
```

**📤 실행 결과:**

```
<Figure size 360x360 with 1 Axes>
```

---

### 📝 셀 14 - 마크다운

### 최적의 K 값 찾기 (엘보우 방식 Elbow Method)

---

### 💻 셀 15 - 코드

```python
from sklearn.cluster import KMeans
inertia_list = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(X)
    inertia_list.append(kmeans.inertia_) # 각 지점으로부터 클러스터의 중심(centroid) 까지의 거리의 제곱의 합
    
plt.plot(range(1, 11), inertia_list)
plt.title('Elbow Method')
plt.xlabel('n_clusters')
plt.ylabel('inertia')
plt.show()
```

**📤 실행 결과:**

```
<Figure size 432x288 with 1 Axes>
```

---

### 📝 셀 16 - 마크다운

### 최적의 K (4) 값으로 KMeans 학습

---

### 💻 셀 17 - 코드

```python
K = 4 # 최적의 K 값
```

---

### 💻 셀 18 - 코드

```python
kmeans = KMeans(n_clusters=K, random_state=0)
# kmeans.fit(X)
y_kmeans = kmeans.fit_predict(X)
```

---

### 💻 셀 19 - 코드

```python
y_kmeans
```

**📤 실행 결과:**

```
array([2, 3, 3, 0, 0, 1, 1, 0, 2, 0, 0, 3, 1, 3, 3, 0, 1, 2, 3, 0, 1, 0,
       3, 1, 2, 2, 3, 3, 3, 3, 1, 1, 3, 0, 2, 2, 3, 0, 0, 0, 3, 1, 2, 3,
       3, 2, 1, 0, 1, 1, 2, 0, 1, 1, 0, 0, 0, 0, 3, 1, 1, 2, 2, 2, 2, 1,
       1, 0, 1, 2, 3, 2, 2, 2, 3, 3, 3, 3, 0, 2, 1, 2, 1, 1, 2, 0, 3, 1,
       2, 3, 0, 1, 0, 2, 3, 2, 2, 0, 1, 3])
```

---

### 📝 셀 20 - 마크다운

### 데이터 시각화 (최적의 K)

---

### 💻 셀 21 - 코드

```python
centers = kmeans.cluster_centers_ # 클러스터의 중심점 (centroid) 좌표
centers
```

**📤 실행 결과:**

```
array([[-0.57163957,  0.85415973],
       [ 0.8837666 , -1.26929779],
       [ 0.94107583,  0.93569782],
       [-1.22698889, -0.46768593]])
```

---

### 💻 셀 22 - 코드

```python
for cluster in range(K):
    plt.scatter(X[y_kmeans == cluster, 0], X[y_kmeans == cluster, 1], s=100, edgecolor='black') # 각 데이터
    plt.scatter(centers[cluster, 0], centers[cluster, 1], s=300, edgecolor='black', color='yellow', marker='s') # 중심점 네모
    plt.text(centers[cluster, 0], centers[cluster, 1], cluster, va='center', ha='center') # 클러스터 텍스트 출력
    
plt.title('Score by hours')
plt.xlabel('hours')
plt.ylabel('score')
plt.show()
```

**📤 실행 결과:**

```
<Figure size 432x288 with 1 Axes>
```

---

### 📝 셀 23 - 마크다운

### 데이터 시각화 (스케일링 원복)

---

### 💻 셀 24 - 코드

```python
X_org = sc.inverse_transform(X) # Feature Scaling 된 데이터를 다시 원복
X_org[:5]
```

**📤 실행 결과:**

```
array([[ 7.33, 73.  ],
       [ 3.71, 55.  ],
       [ 3.43, 55.  ],
       [ 3.06, 89.  ],
       [ 3.33, 79.  ]])
```

---

### 💻 셀 25 - 코드

```python
centers_org = sc.inverse_transform(centers)
centers_org
```

**📤 실행 결과:**

```
array([[ 3.96458333, 76.08333333],
       [ 7.8552    , 20.96      ],
       [ 8.0084    , 78.2       ],
       [ 2.21269231, 41.76923077]])
```

---

### 💻 셀 26 - 코드

```python
for cluster in range(K):
    plt.scatter(X_org[y_kmeans == cluster, 0], X_org[y_kmeans == cluster, 1], s=100, edgecolor='black') # 각 데이터
    plt.scatter(centers_org[cluster, 0], centers_org[cluster, 1], s=300, edgecolor='black', color='yellow', marker='s') # 중심점 네모
    plt.text(centers_org[cluster, 0], centers_org[cluster, 1], cluster, va='center', ha='center') # 클러스터 텍스트 출력
    
plt.title('Score by hours')
plt.xlabel('hours')
plt.ylabel('score')
plt.show()
```

**📤 실행 결과:**

```
<Figure size 432x288 with 1 Axes>
```

---


## 📓 06. Quiz.ipynb

> **파일 경로**: `C:\pythonworkspace\06. Quiz.ipynb`
> **총 셀 개수**: 21개

### 📝 셀 1 - 마크다운

# 6. Quiz

---

### 📝 셀 2 - 마크다운

## 어느 결혼식장에서 피로연의 식수 인원을 올바르게 예측하지 못하여 버려지는 음식으로 고민이 많다고 합니다. 현재까지 진행된 결혼식에 대한 결혼식 참석 인원과 그 중에서 식사를 하는 인원의 데이터가 제공될 때, 아래 각 문항에 대한 코드를 작성하시오.

---

### 📝 셀 3 - 마크다운

주의) 사전 작업으로 아래 코드 셀을 먼저 실행하시오

---

### 💻 셀 4 - 코드

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

---

### 📝 셀 5 - 마크다운

## 1) QuizData.csv 파일로부터 데이터를 읽어와서 결혼식 참석 인원(total), 식수 인원(reception)을 각각의 변수로 저장하시오.

---

### 💻 셀 6 - 코드

```python
dataset = pd.read_csv('QuizData.csv')
dataset[:5]
```

**📤 실행 결과:**

```
total  reception
0    118         62
1    253        148
2    320        201
3     94         80
4    155         92
```

---

### 💻 셀 7 - 코드

```python
X = dataset.iloc[:, :-1].values # 결혼식 참석 인원 total
y = dataset.iloc[:, -1].values # 식수 인원 reception
```

---

### 💻 셀 8 - 코드

```python
X[:5], y[:5]
```

**📤 실행 결과:**

```
(array([[118],
        [253],
        [320],
        [ 94],
        [155]], dtype=int64),
 array([ 62, 148, 201,  80,  92], dtype=int64))
```

---

### 📝 셀 9 - 마크다운

## 2) 전체 데이터를 훈련 세트와 테스트 세트로 분리하시오. 이 때 비율은 75 : 25 로 합니다.

(단, random_state = 0 으로 설정)

---

### 💻 셀 10 - 코드

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
```

---

### 📝 셀 11 - 마크다운

## 3) 훈련 세트를 이용하여 단순 선형 회귀 (Simple Linear Regression) 모델을 생성하시오.

---

### 💻 셀 12 - 코드

```python
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
```

**📤 실행 결과:**

```
LinearRegression()
```

---

### 📝 셀 13 - 마크다운

## 4) 데이터 시각화 (훈련 세트) 코드를 작성하시오.

---

### 💻 셀 14 - 코드

```python
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, reg.predict(X_train), color='green')
plt.title('Wedding reception (train)')
plt.xlabel('total')
plt.ylabel('reception')
plt.show()
```

**📤 실행 결과:**

```
<Figure size 432x288 with 1 Axes>
```

---

### 📝 셀 15 - 마크다운

## 5) 데이터 시각화 (테스트 세트) 코드를 작성하시오.

---

### 💻 셀 16 - 코드

```python
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_train, reg.predict(X_train), color='green')
plt.title('Wedding reception (test)')
plt.xlabel('total')
plt.ylabel('reception')
plt.show()
```

**📤 실행 결과:**

```
<Figure size 432x288 with 1 Axes>
```

---

### 📝 셀 17 - 마크다운

## 6) 훈련 세트, 테스트 세트에 대해 각각 모델 평가 점수를 구하시오.

---

### 💻 셀 18 - 코드

```python
reg.score(X_train, y_train) # 훈련 세트 평가 점수
```

**📤 실행 결과:**

```
0.8707088403321211
```

---

### 💻 셀 19 - 코드

```python
reg.score(X_test, y_test) # 테스트 세트 평가 점수
```

**📤 실행 결과:**

```
0.8634953212566615
```

---

### 📝 셀 20 - 마크다운

## 7) 결혼식 참석 인원이 300명일 때 예상되는 식수 인원을 구하시오.

---

### 💻 셀 21 - 코드

```python
total = 300 # 결혼식 참석 인원
y_pred = reg.predict([[total]])

print(f'결혼식 참석 인원 {total} 명에 대한 예상 식수 인원은 {np.around(y_pred[0]).astype(int)} 명입니다.')
```

**📤 실행 결과:**

```
결혼식 참석 인원 300 명에 대한 예상 식수 인원은 177 명입니다.
```

---


---

## 📊 분석 요약

- **총 노트북 파일**: 6개
- **총 셀 개수**: 172개
- **코드 셀**: 119개
- **마크다운 셀**: 53개

## 🔗 유용한 링크

- [Jupyter Notebook 공식 문서](https://jupyter-notebook.readthedocs.io/)
- [Python 공식 문서](https://docs.python.org/)
- [Notion 마크다운 가이드](https://www.notion.so/help/writing-and-editing-basics)

---

✨ **자동 생성됨** - Jupyter to Notion 변환기 v1.0
