import numpy as np
import matplotlib.pyplot as plt

def show_sigmoid():
    def sigmoid(z):
        return 1 / (1 + np.e ** -z)

    print(np.e)
    print()

    print(sigmoid(-1))
    print(sigmoid(0))
    print(sigmoid(1))
    print()

    for z in np.arange(-5, 5, 0.1):
        s = sigmoid(z)
        print(z, s)

        plt.plot(z, s, 'ro')
    plt.show()


def show_selection():
    def log_a():
        return 'A'

    def log_b():
        return 'B'

    y = 1
    print(y * log_a() + (1 - y) + log_b())

    if y == 1:
        print(log_a())
    else:
        print(log_b())

    y = 0
    print(y * log_a() + (1 - y) + log_b())
    if y == 1:
        print(log_a())
    else:
        print(log_b())

# x축 즉, 시그모리 범위는 0과 1 사이
# https://www.desmos.com/calculator
# 보라색 선을
# sigmoid 가 0.4
# 바이너리 크로스엔트로피 : 두 개의 최적의 값은 겹친점

# show_sigmoid()
show_selection()
