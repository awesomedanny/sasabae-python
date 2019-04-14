# 변수가 여려개 있음
import tensorflow as tf
import numpy as np

#문제
#아래 코드가 동작하도록 수정
def multiple_regression_1():
    # y = x1 + x2
    # hx = w1 * x1 + w2 * x2 + b
    #       1         1        0
    x1 = [1, 0, 3, 0, 5] #공부한 시간 - 피처 1
    x2 = [0, 2, 0, 4, 0] #출석한 일수 - 피처 2
    y = [1, 2, 3, 4, 5] #성적

    w1 = tf.Variable(tf.random_uniform([1])) #난수, 2차원[2, 4]
    w2 = tf.Variable(tf.random_uniform([1])) #난수, 2차원[2, 4]
    b = tf.Variable(tf.random_uniform([1]))


    hx = w1 * x1 + w2 * x2 + b
    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i) #cost 계산.. 평균값 3차원->2차원?

    optimizer = tf.train.GradientDescentOptimizer(0.1) #미분함, 기울기0.1, learning_rate는 생략가능
    train = optimizer.minimize(loss) #loss가 줄어드는 방향  train은 연결하여 사용한다의미

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train) #연산시작

        c = sess.run(loss)
        print(i, c)


    sess.close()

# 피처가 늘어났다고 x 변수를 두개를 쓰는건 불가
def multiple_regression_2():
    x = [[1, 0, 3, 0, 5], #공부한 시간 - 피처 1
         [0, 2, 0, 4, 0]] #출석한 일수 - 피처 2
    y = [1, 2, 3, 4, 5] #성적

    w = tf.Variable(tf.random_uniform([2])) #난수, 2차원[2, 4]
    b = tf.Variable(tf.random_uniform([1]))


    hx = w[0] * x[0] + w[1] * x[1] + b  #이 부분은 '행렬곱셈'으로 처리해야
    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i) #cost 계산.. 평균값 3차원->2차원?

    optimizer = tf.train.GradientDescentOptimizer(0.1) #미분함, 기울기0.1, learning_rate는 생략가능
    train = optimizer.minimize(loss) #loss가 줄어드는 방향  train은 연결하여 사용한다의미

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train) #연산시작

        c = sess.run(loss)
        print(i, c)

    sess.close()

# 행렬곱셈으로 바꾸세요 (tf.matmul)
def multiple_regression_3():
    x = [[1., 0., 3., 0., 5.], #공부한 시간 - 피처 1
         [0., 2., 0., 4., 0.]] #출석한 일수 - 피처 2
    y = [1, 2, 3, 4, 5] #성적

    w = tf.Variable(tf.random_uniform([1, 2])) #난수, 2차원[2, 4] : 2행 4열 의미
    b = tf.Variable(tf.random_uniform([1]))


    # () = (1, 2) @ (2, 5)
    hx = tf.matmul(w, x) + b
    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i) #cost 계산.. 평균값 3차원->2차원?

    optimizer = tf.train.GradientDescentOptimizer(0.1) #미분함, 기울기0.1, learning_rate는 생략가능
    train = optimizer.minimize(loss) #loss가 줄어드는 방향  train은 연결하여 사용한다의미

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train) #연산시작

        c = sess.run(loss)
        print(i, c)


    sess.close()

#문제 : bias 변수 b를 w에 넣어주세요.
def multiple_regression_4():
    # x = [[1., 0., 3., 0., 5.], #공부한 시간 - 피처 1
    #      [0., 2., 0., 4., 0.], #출석한 일수 - 피처 2
    #      [1., 1., 1., 1., 1.]]  #===========> bias 추가 위치에는 상관없다

    #bias의 위치 중요
    x = [[1., 1., 1., 1., 1.], #===========> bias 추가 위치에는 상관없다
         [1., 0., 3., 0., 5.], #공부한 시간 - 피처 1
         [0., 2., 0., 4., 0.]] #출석한 일수 - 피처 2

    y = [1, 2, 3, 4, 5] #성적

    # x = tf.constant([1., 0., 3., 0., 5., 0., 2., 0., 4., 5.], shape=[2, 5])
    # y = tf.constant([1, 2, 3, 4, 5], shape=[1, 5])

    w = tf.Variable(tf.random_uniform([1, 3])) #난수, 2차원[1, 3] 3열로 만들어줌

    # (1, 5) = (1, 3) @ (3, 5)
    # hx = w[0] * x[0] + w[1] * x[1] + b   브로드캐스팅
    # hx = w[0] * x[0] + w[1] * x[1] + w[2] + x[2]   배열+배열=벡터연산
    hx = tf.matmul(w, x)
    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i) #cost 계산.. 평균값 3차원->2차원?

    optimizer = tf.train.GradientDescentOptimizer(0.1) #미분함, 기울기0.1, learning_rate는 생략가능
    train = optimizer.minimize(loss) #loss가 줄어드는 방향  train은 연결하여 사용한다의미

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train) #연산시작
        c = sess.run(loss)
        print(i, c)

    print(sess.run(w)) #마지막 값이 bias
    sess.close()

#문제 : 3시간 공부하 6번 출석한 학생과
#       5시간 공부하고 9번 출석한 학생의 성적을 알려주세요.
def multiple_regression_5():
    x = [[1., 1., 1., 1., 1.], #bias 추가 위치에는 상관없다
         [1., 0., 3., 0., 5.], #공부한 시간 - 피처 1
         [0., 2., 0., 4., 0.]] #출석한 일수 - 피처 2

    y = [1, 2, 3, 4, 5] #성적

    w = tf.Variable(tf.random_uniform([1, 3])) #난수, 2차원[1, 3] 3열로 만들어줌

    # 플레이스홀더
    ph_x = tf.placeholder(tf.float32)


    hx = tf.matmul(w, ph_x)
    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i) #cost 계산.. 평균값 3차원->2차원?

    optimizer = tf.train.GradientDescentOptimizer(0.1) #미분함, 기울기0.1, learning_rate는 생략가능
    train = optimizer.minimize(loss) #loss가 줄어드는 방향  train은 연결하여 사용한다의미

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, {ph_x: x})
        c = sess.run(loss, {ph_x: x})
        # _, c = sess.run([train, loss], {ph_x: x})

        print(i, c)

    print('x=1 2 3 :', sess.run(hx, {ph_x: [[1.],
                                            [3.],
                                            [6.]]}))
    print('x=1 2 3 :', sess.run(hx, {ph_x: [[1.],
                                            [5.],
                                            [9.]]}))
    print('x=1 2 3 :', sess.run(hx, {ph_x: x})) #hx의 최적의 값 x
    print('x=1 2 3 :', sess.run(hx, {ph_x: [[1., 1.],
                                            [3., 5.],
                                            [4., 9.]]}))

    sess.close()

# 문제 : 4번코드에서 행렬 곱셈의 순서를 바꾸세요. x가 앞에 오도록
def multiple_regression_6():
    x = [[1., 1., 0.],
         [1., 0., 2.],
         [1., 3., 0.],
         [1., 0., 4.],
         [1., 5., 0.]]

    y = [[1], [2], [3], [4], [5]] #성적도 2차원으로 맞춰줘야 함

    # 모델 : 이 부분만 바꾸며
    w = tf.Variable(tf.random_uniform([3, 1])) #난수, 2차원[1, 3] 3열로 만들어줌
    # (5, 1) = (5, 3) @ (3, 1)
    hx = tf.matmul(x, w) #hx:하이퍼써제스

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i) #cost 계산.. 평균값 3차원->2차원?

    optimizer = tf.train.GradientDescentOptimizer(0.1) #미분함, 기울기0.1, learning_rate는 생략가능
    train = optimizer.minimize(loss) #loss가 줄어드는 방향  train은 연결하여 사용한다의미

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train) #연산시작
        c = sess.run(loss)
        print(i, c)

    print(sess.run(w)) #마지막 값이 bias
    sess.close()


# 다운로드 : https://vincentarelbundock.github.io/Rdatasets/datasets.html
#문제 :
# Girth가 10이고 Height가 60일때와
# Girth가 15이고 Height가 80일 때의 볼륨을 구하세요.
def multiple_regression_trees_1():
    trees = np.loadtxt('Data/trees.csv', delimiter=',', skiprows=1, unpack=True)
    print(trees.shape)
    print(type(trees))

    x = [trees[0], trees[1]]
    y = trees[2]

    # 모델 : 이 부분만 바꾸며
    w = tf.Variable(tf.random_uniform([1, 2])) #난수, 2차원[1, 3] 3열로 만들어줌
    b = tf.Variable(tf.random_uniform([1])) #난수, 2차원[1, 3] 3열로 만들어줌

    # 플레이스홀더
    ph_x = tf.placeholder(tf.float32)

    # (1, 31) = (1, 2) @ (2, 31)
    hx = tf.matmul(w, ph_x) + b

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.0001) #0.001 가 기본값
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, {ph_x: x})
        c = sess.run(loss, {ph_x: x})
        print(i, c)

    print('x=30 50 :', sess.run(hx, {ph_x: [[10,15],[60,80]]}))
    sess.close()


def multiple_regression_trees_2():
    trees = np.loadtxt('Data/trees.csv', delimiter=',', skiprows=1, unpack=True)
    print(trees.shape)
    print(type(trees))

    x = [[1.] * len(trees[0]),  # np.ones(31) 과 동일한 의미
         trees[0],
         trees[1]]
    y = trees[2]

    # 모델 : 이 부분만 바꾸며
    w = tf.Variable(tf.random_uniform([1, 3])) #난수, 2차원[1, 3] 3열로 만들어줌

    # 플레이스홀더
    ph_x = tf.placeholder(tf.float32)

    # (1, 31) = (1, 3) @ (3, 31)
    hx = tf.matmul(w, ph_x)

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.0001) #0.001 가 기본값
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, {ph_x: x})
        c = sess.run(loss, {ph_x: x})
        print(i, c)

    print('x=30 50 :', sess.run(hx, {ph_x: [[1, 1], [10, 15], [60, 80]]}))
    sess.close()


def multiple_regression_trees_3():
    trees = np.loadtxt('Data/trees.csv', delimiter=',', skiprows=1)
    print(trees.shape)

    x = trees[:, :-1]
    y = trees[:, -1:] #행렬 까지 보존

    # 모델 : 이 부분만 바꾸며
    w = tf.Variable(tf.random_uniform([2, 1]))
    b = tf.Variable(tf.random_uniform([1]))

    # 플레이스홀더
    ph_x = tf.placeholder(tf.float32)

    # (31, 1) = (31, 2) @ (2, 1)
    hx = tf.matmul(ph_x, w) + b

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.0001) #0.001 가 기본값
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, {ph_x: x})
        c = sess.run(loss, {ph_x: x})
        print(i, c)

    print('x=30 50 :', sess.run(hx, {ph_x: [[10, 60], [15, 70]]}))
    sess.close()



#모두의 딥러닝 동영상강의 ML/DL for EVERYONE season2

# multiple_regression_1()
# multiple_regression_2()
# multiple_regression_3()
# multiple_regression_4()
# multiple_regression_5()
# multiple_regression_6()
# multiple_regression_trees_1()
# multiple_regression_trees_2()
multiple_regression_trees_3()





