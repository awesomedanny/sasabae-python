# Day_01_01_python.py

# ctrl + shift + f10
# tab  or shift + tab
# ctrl + / 주석처리

# tensorflow, matplotlib
# H(x) = W(x) + b  =>  y:타겟 또는 레이블, x:피처
# import tensorflow
import numpy as np
import matplotlib.pyplot as plt

def cost(x, y, w):
    loss = 0
    for i in range(len(x)):
        hx = w * x[i]
        loss += (hx - y[i]) ** 2

    return loss / len(x)


def gradient_descent(x, y, w):
    loss = 0
    for i in range(len(x)):
        hx = w * x[i]
        loss += (hx - y[i]) * x[i]

    return loss / len(x)


def show_cost():
    # y = x 1
    # H(x) = W(x) + b
    #        1      0
    x = [1, 2, 3]
    y = [1, 2, 3]

    print(cost(x, y, 2.0))
    print(cost(x, y, 1.0))
    print(cost(x, y, 0.0))
    print('-' * 50)

    for w in np.arange(-3, 5, 0.1):
        c = cost(x, y, w)
        print('{:5.2f} {:5.2f}'.format(w, c))

        plt.plot(w, c, 'ro')

    plt.xlabel('w')
    plt.ylabel('cost')
    plt.show()

# w를 1.0으로 만들 수 있는 방법 3가지를 찾으세요.
# w = 1,  range를 60 이상
# 1. range를 100으로 훈련횟수 증가
# 2. 학습률 0.1 을 0.21
# 3. 초기값

# 문제
# x가 5와 7일 때의 y를 구하세요.
def show_gradient():
    x = [1, 2, 3]
    y = [1, 2, 3]
    w = 5
    for i in range(100):
        c = cost(x, y, w)
        g = gradient_descent(x, y, w)
        w -= 0.1 * g
        print(i, c)

        # 조기종료(early stopping)
        if c < 1e-15:
            break
    print('5 :', w * 5)
    print('7 :', w * 7)

#실행----------------------------------------
# show_cost()
show_gradient()


# 미분 : 기울기, 순간변화량
#        x축으로 1만큼 이동했을 때 y축으로 이동한 거리

# y = 3       3=1, 3=2, 3=3 .. 0 x는 미영향(미분상수)
# y = x       1=1, 2=2, 3=3 .. 1
# y = 2x      2=1, 4=2, 6=3 .. 2
# y = x+1     2=1, 3=2, 4=3 .. 1  +1바이오스
# y = xz

# y = x^2     1=1, 4=2, 9=3  --> 2*x^(2-1) = 2x
# 미분을 하려면                   2*x^(2-1) * x미분
# y = (x+1)^2                --> 2*(x+1)^(2-1) = 2(x+1)
# 미분을 하려면                   2*(x+1)^(2-1) * (x+1)미분














