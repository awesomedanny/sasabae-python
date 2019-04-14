# Day_01_02_linearRegression.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# tf.enable_eager_execution()  #
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(w)
# print(sess.run(w))

# 문제 : x가 5와 7일때 y값
def linear_regression_1():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = tf.Variable(10.0)
    b = tf.Variable(5.0)


    # 케이스1
    # hx = tf.add(tf.multiply(w, x), b) #w*x+b
    # loss_i = tf.square(hx - y) #제곱

    # 케이스1 = 케이스2
    hx = w * x + b  #곱셈을 3번, 결과에 바이오스3번더함(브로드캐스팅) 결국 hx 는 배열
    loss_i = (hx - y) ** 2 #벡터연산

    loss = tf.reduce_mean(loss_i) #cost 계산.. 평균값 3차원->2차원?

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1) #미분함, 기울기0.1, learning_rate는 생략가능
    train = optimizer.minimize(loss=loss) #loss가 줄어드는 방향  train은 연결하여 사용한다의미


    with tf.Session() as sess:  #with는 close() 포함
        sess.run(tf.global_variables_initializer())

        for i in range(10):
            sess.run(train)

            c = sess.run(loss)
            print(i, c)


        print('5:', sess.run(w*5+b))
        print('7:', sess.run(w*7+b))

        ww, bb = sess.run([w, b])
        print(ww, bb)

        print('5:', ww*5+bb)
        print('7:', ww*7+bb)


def linear_regression_2():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = tf.Variable(tf.random_uniform([1])) #난수, 2차원[2, 4]
    b = tf.Variable(tf.random_uniform([1]))

    # 플레이스홀더
    ph_x = tf.placeholder(tf.float32)
    ph_y = tf.placeholder(tf.float32)

    hx = w * ph_x + b
    loss_i = (hx - ph_y) ** 2
    loss = tf.reduce_mean(loss_i) #cost 계산.. 평균값 3차원->2차원?

    optimizer = tf.train.GradientDescentOptimizer(0.1) #미분함, 기울기0.1, learning_rate는 생략가능
    train = optimizer.minimize(loss) #loss가 줄어드는 방향  train은 연결하여 사용한다의미

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, feed_dict={ph_x: x, ph_y: y}) #연산시작

        c = sess.run(loss, {ph_x: x, ph_y: y})
        print(i, c)

    print('-' * 50)
    print('x=5 :', sess.run(hx, {ph_x: 5}))
    print('x=7 :', sess.run(hx, {ph_x: 7}))
    print('x=7 :', sess.run(hx, {ph_x: x}))
    print('x=1 2 3 :', sess.run(hx, {ph_x: [1, 2, 3]}))
    print('x=1 2 3 :', sess.run(hx, {ph_x: [5, 7]}))
    sess.close()

# 문제
# _2() 코드에서 잘못된 부분을 수정하세요
def linear_regression_3():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = tf.Variable(tf.random_uniform([1]))
    b = tf.Variable(tf.random_uniform([1]))

    # 플레이스홀더
    ph_x = tf.placeholder(tf.float32)

    hx = w * ph_x + b
    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, feed_dict={ph_x: x})

        c = sess.run(loss, {ph_x: x})
        print(i, c)

    print('-' * 50)
    print('x=5 :', sess.run(hx, {ph_x: 5}))
    print('x=7 :', sess.run(hx, {ph_x: 7}))
    print('x=7 :', sess.run(hx, {ph_x: x}))
    print('x=1 2 3 :', sess.run(hx, {ph_x: [1, 2, 3]}))
    print('x=1 2 3 :', sess.run(hx, {ph_x: [5, 7]}))
    sess.close()

# 문제
# 속도가 30과 50일때의 제동거리를 알려주세요.
def linear_regression_cars():
    #np 는 보통 숫자
    cars = np.loadtxt('Data/cars.csv', delimiter=',', unpack=True) #행과 열을 바꿈 전치
    print(cars)
    print(cars.shape)

    x, y = cars

    w = tf.Variable(tf.random_uniform([1])) #0~1
    b = tf.Variable(tf.random_uniform([1]))

    # 플레이스홀더
    ph_x = tf.placeholder(tf.float32)

    hx = w * ph_x + b
    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.001) #0.001 가 기본값
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, feed_dict={ph_x: x})

        c = sess.run(loss, {ph_x: x})
        print(i, c)

    print('-' * 50)
    print('x=30 50 :', sess.run(hx, {ph_x: [30,50]}))
    y0, y1, y2 = sess.run(hx, {ph_x: [0, 30, 50]})  #바이오스가 있어야 함 0
    sess.close()


    plt.plot(x, y, 'ro')
    plt.plot([0, 30], [0, y1], 'g')
    plt.plot([0, 30], [y0, y1], 'k')
    plt.plot([0, 30, 50], [y0, y1, y2], 'm')
    plt.show()


# linear_regression_1()
# linear_regression_2()
# linear_regression_3()
linear_regression_cars()


# data의 csv에서 ctrl_r 눌러 "\d*", 를 replace all 로 제거


