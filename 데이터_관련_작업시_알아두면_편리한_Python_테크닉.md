# 데이터 관련 작업시 알아두면 편리한 Python 테크닉



<a href="https://colab.research.google.com/github/serithemage/DataScienctPractice/blob/main/Python_%EC%A4%91%EA%B8%89_%ED%85%8C%ED%81%AC%EB%8B%89.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# 1. 리스트 내포(List comprehension)  
제가 처음 리스트 내포 표현식을 봤을때 그야말로 충격과 공포 그 자체였습니다. '배열 괄호 안에 for문이 들어가 있네?! 이거 너무 한거 아냐?' 사실 바로 다음에 다룰 람다 표현식도 비슷한 느낌이었는데 익숙해 지고 나면 다 별거 아닙니다.  
암튼 이 리스트 내포 표현식은 리스트 내부에서 반복 연산을 편리하게 수행할 수 있게 해 줍니다.

```python
import numpy as np
```

우선 1부터 100 시이의 랜덤 정수를 100개 생성해서 배열에 넣어봅시다.

```python
data = np.random.randint(1,100,100)
```

잘 들어갔는지 이제 내용을 확인해 봅시다.

```python
data
```




    array([27, 79, 50, 63, 30, 69, 61, 34, 53, 67, 58, 32, 58, 70, 11, 14, 22,
           69, 75, 20, 38, 52, 37, 69, 28, 75, 91, 73, 95, 31, 82, 35, 64, 70,
           58, 52, 55, 99, 59, 18, 93, 51, 73, 88, 16, 76, 95, 65, 86, 44, 92,
           46, 93, 40, 85, 53, 87, 39, 45,  8, 95, 18, 39, 62, 47, 47, 38, 64,
           24, 25, 51, 71, 57, 55, 29, 10, 26, 83,  9, 34,  3, 41,  5, 14, 25,
           76, 39, 53, 36,  4,  5, 17, 77, 35, 70, 21, 89, 91, 27, 92])



배열같기는 한데 약간 다른걸 알 수 있습니다. 오브젝트 타입을 확인해 봅시다.

```python
type(data)
```




    numpy.ndarray



numpy의 ndarray(n차원을 다루기위한 데이터 형)이라는 것을 알 수 있습니다. 사용법은 일반 배열과 크게 다르지 않습니다.

이제 길이를 확인해 봅시다.

```python
len(data)
```




    100



예상대로 1000개가 들어 있네요.  
이번에는 리스트 내포 표기를 사용해서 배열 안의 모든 값에 10을 곱해 봅시다. 

```python
 data_10x = [ value * 10 for value in data ]
```

어떼요? 참 쉽죠?  
잘 곱해졌는지 내용을 확인해 봅시다.

```python
data_10x
```




    [270,
     790,
     500,
     630,
     300,
     690,
     610,
     340,
     530,
     670,
     580,
     320,
     580,
     700,
     110,
     140,
     220,
     690,
     750,
     200,
     380,
     520,
     370,
     690,
     280,
     750,
     910,
     730,
     950,
     310,
     820,
     350,
     640,
     700,
     580,
     520,
     550,
     990,
     590,
     180,
     930,
     510,
     730,
     880,
     160,
     760,
     950,
     650,
     860,
     440,
     920,
     460,
     930,
     400,
     850,
     530,
     870,
     390,
     450,
     80,
     950,
     180,
     390,
     620,
     470,
     470,
     380,
     640,
     240,
     250,
     510,
     710,
     570,
     550,
     290,
     100,
     260,
     830,
     90,
     340,
     30,
     410,
     50,
     140,
     250,
     760,
     390,
     530,
     360,
     40,
     50,
     170,
     770,
     350,
     700,
     210,
     890,
     910,
     270,
     920]



data_10x가 어떠한 자료형인지 확인해 봅시다.

```python
type(data_10x)
```




    list



ndarray가 아니라 list로 바뀐것을 볼 수 있습니다. 즉 data_10x는 기존의 data의 값을 가져다 만들긴 했어도 리스트 내포 표기를 이용해 만들어진 새로운 배열입니다.

그 다음은 데이터에서 짝수만 뽑아봅시다.  
다음과 같이 리스트 내포 표기를 사용하면 아주 간단합니다.

```python
odd_data = [e for e in data if e % 2 == 0]
```

잘 들어갔는지 내용을 확인해 봅시다.


```python
odd_data
```




    [50,
     30,
     34,
     58,
     32,
     58,
     70,
     14,
     22,
     20,
     38,
     52,
     28,
     82,
     64,
     70,
     58,
     52,
     18,
     88,
     16,
     76,
     86,
     44,
     92,
     46,
     40,
     8,
     18,
     62,
     38,
     64,
     24,
     10,
     26,
     34,
     14,
     76,
     36,
     4,
     70,
     92]



if가 된다면 else도 가능하지 않을까요?  
잘 알려진 문제인 [FizzBuzz 문제](https://en.wikipedia.org/wiki/Fizz_buzz)를 내포 표기로 풀어봅시다.

문법은  
 
> [ **반환값** if **조건식** else **반환값** if ... else **기본 반환값** for i in **배열** ] 

입니다.  

처음엔 익숙치 않아 헤깔리지만 FizzBuzz정도만 연습해서 혼자 작성할 수 있게 된다면 실무에서 사용하는데에도 크게 문제 없습니다. 반환값이 if문 앞에 온다는 점과 for 문이 맨 뒤에 붙는다는 점이 특이하네요.

```python
fizz_buzz = ['FizzBuzz' if i%3 == 0 and i%5 == 0 else 'Fizz' if i%3 == 0 else 'Buzz' if i%5 == 0 else i for i in range(1,51)]
```

```python
fizz_buzz
```




    [1,
     2,
     'Fizz',
     4,
     'Buzz',
     'Fizz',
     7,
     8,
     'Fizz',
     'Buzz',
     11,
     'Fizz',
     13,
     14,
     'FizzBuzz',
     16,
     17,
     'Fizz',
     19,
     'Buzz',
     'Fizz',
     22,
     23,
     'Fizz',
     'Buzz',
     26,
     'Fizz',
     28,
     29,
     'FizzBuzz',
     31,
     32,
     'Fizz',
     34,
     'Buzz',
     'Fizz',
     37,
     38,
     'Fizz',
     'Buzz',
     41,
     'Fizz',
     43,
     44,
     'FizzBuzz',
     46,
     47,
     'Fizz',
     49,
     'Buzz']



# 2. Lambda 함수
Lambda함수는 간단한 처리를 익명 함수로 만들어서 처리에 사용하도록 마치 데이터처럼 전달하는것이 가능한 함수입니다.
특히 배열에 대해서 여러가지 작업을 수행하고자 할 때, map, reduce, filter와 같은 함수와 함께 Lambda 함수를 이용하면 매우 깔끔하게 작업 내용을 기술할 수가 있습니다.

예를 들어 다음과 같은 삼각형의 넓이를 계산하는 함수가 있다고 가정해 봅시다.

```python
def calc(base, height):
  return base*height/2

calc(5,10)
```




    25.0



이를 람다 함수로 표현하면 다음과 같습니다.

```python
(lambda base, height: base*height/2)(5,10)
```




    25.0



재사용 가능하도록 func라는 이름으로 람다 함수를 선언해 보죠.

```python
func = lambda base, height: base*height/2
```

```python
func(5,10)
```




    25.0



이제 map, reduce, filter와 함께 사용하는 방버을 살펴보겠습니다.

우선 이름을 넣은 배열을 하나 작성해 보겠습니다.

```python
names = ['Jung', 'Kim', 'Park', 'Choi']
```

이 배열의 내용을 전부 대문자로 변환해 봅시다. for문을 사용한다면 다음과 같이 기술할 수 있을겁니다.

```python
upper_case_names1 = []
for name in names:
  upper_case_names1.append(name.upper())

upper_case_names1
```




    ['JUNG', 'KIM', 'PARK', 'CHOI']



같은 내용을 위에서 배운 배열의 내포 표기를 사용해서 적으면 다음과 같습니다.

```python
upper_case_names2 = [name.upper() for name in names]
upper_case_names2
```




    ['JUNG', 'KIM', 'PARK', 'CHOI']



같은 내용을 map과 lambda함수를 사용하면 다음과 같이 적을 수 있습니다.

```python
upper_case_name3 = list(map(lambda name: name.upper(), names))
upper_case_name3
```




    ['JUNG', 'KIM', 'PARK', 'CHOI']



이 lambda 식을 이용한 map은 list뿐만 아니라 Pandas DataFrame에도 적용이 가능하다. DataFrame에서의 map사용에 대해서는 [Pandas의 map함수, apply함수, applymap함수 차이점 분석](http://www.leejungmin.org/post/2018/04/21/pandas_apply_and_map/) 이라는 블로그에 잘 정리 되어 있다.
