import numpy as np

a = np.arange(6) #array range
print(a)
print(a.shape, a.dtype) #4바이트정수

b = a.reshape(2, 3)
print(b)
print(b.shape, b.dtype) #4바이트정수
print('-' * 50)

print(a + 1)    #각 배열값에 1을 더함 : 브로트캐스팅
print(b + 1)
print(a + a)
print(b + b)
print('-' * 50)

# print(a + b) #1차원 + 2차원이므로 에러

a1 = np.arange(3).reshape(1, 3)
a2 = np.arange(3).reshape(3, 1)
a3 = np.arange(6)
a4 = np.arange(6).reshape(1, 6)
# print(a1 + a3)  #error
print(a2 + a3)  #no error
print(a2 + a4)  #error
print(a2 + a2)  #error
print('-' * 50)

print(a)
print(a[0], a[1])
print(a[-1], a[-2])
print(" : ", a[2:4])
print(" : ", a[:4])
print(" : ", a[2:])
print('-' * 50)

print(a[:])
print(a[::2])
print(a[3:4])
print(a[3:3])  #같을 때 비어있다로 표현

#a를 거꾸로 출력
print(a[5:-1:-1])
print(a[::-1])

print('-' * 50)

c = np.arange(15).reshape(-1, 5)
print('c', c)
print(c[:2])
print(c[::-1])
print(c[::-1][::-1])

print('-' * 50)

print(c[-1][-1])
print(c[-1, -1])
print(c[::-1, ::-1])

#@문제
#마지막 컬럼만 가져오기
print(c[:, -1])
print(c[:, -1:]) #차원이 유지됨
print('-' * 50)


print(a)
print(a[0], a[1])
print(a[[0, 1]])  #Index Array
print(a[[2, 5, 1, 0, 0, 2]])  #Index Array
k = [2, 5, 1, 0, 0, 2]
print(a[k])

print(a > 2) #[False False False  True  True  True]
print(a[a>2]) #[3 4 5]