import tensorflow as tf
import numpy as np
from sklearn import preprocessing, model_selection


# 문제
# 4시간 공부하고 7번 출석한 학생과
# 9시간 공부하고 출석한 학생의 통과 여부를 알려주세요.
def logistic_regression():
    x = [[1., 1., 1., 1., 1., 1.],
         [2., 3., 6., 5., 8., 9.],  #공부한 시간
         [3., 2., 5., 6., 9., 8.]]  #출석한 일수
    y = [0, 0, 1, 1, 1, 1]
    y = np.int32(y)

    w = tf.Variable(tf.random_uniform([1, 3]))  # 난수, 2차원[1, 3] 3열로 만들어줌

    # 플레이스홀더
    ph_x = tf.placeholder(tf.float32)

    z = tf.matmul(w, ph_x)
    hx = tf.sigmoid(z)
    loss_i = y * -tf.log(hx) + (1-y) * -tf.log(1-hx) #브로드캐스팅 산술식이라서 y가 리스트이므로 y를 변환
    # loss_i = -y * tf.log(hx) - (1-y) * tf.log(1-hx) #브로드캐스팅 산술식이라서 y가 리스트이므로 y를 변환

    loss = tf.reduce_mean(loss_i)  # cost 계산.. 평균값 3차원->2차원?

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, {ph_x: x})
        c = sess.run(loss, {ph_x: x})

        print(i, c)

    print('-' * 50)
    preds = sess.run(hx, {ph_x: [[1., 1.], [4., 9.], [3., 6.]]})
    preds_bool = (preds > 0.5)
    #결과 : [[0.42813823 0.9960498 ]]  ==> 0.5기준 0.42와 0.99
    print(preds_bool)

    preds_idx = np.int32(preds_bool)

    labels = np.array(['fail', 'pass'])
    print(labels[preds_idx])


    sess.close()


def minmax_scale(data):
    mx = np.max(data, axis=0) #수직으로 값을 찾는다
    mn = np.min(data, axis=0)

    print(mx)
    print(mn)

    return (data - mn) / (mx - mn)

# 문제
# 70%의 데이터로 학습하고 30%의 데이터에 대해 성별을 예측하세요.
# 정확도까지 알려주면 좋습니다.
def logistic_regression_horese():
    horses = np.loadtxt('data/HorsePrices.csv', delimiter=',')

    # scipy, scikit-learn, sklearn 모듈 설치
    # 데이터분석, 딥러닝전 머신러닝 라이브러리(뉴럴넷),
    # numpy : 고성능배열라이브러리


    # x = [horses[1], horses[2], horses[3]]
    # y = trees[4]

    x = horses[:, 1:-1]
    y = horses[:, -1:] #행렬 까지 보존
    print(x.shape, y.shape) # (50, 3) (50, 1)

    # x = minmax_scale(x)
    x = preprocessing.minmax_scale(x)

    # train_size = int(len(x) * 0.7)
    # x_train, x_test = x[:train_size], x[train_size:]
    # y_train, y_test = y[:train_size], y[train_size:]

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=35)

    print(x_train.shape, x_test.shape) #(35, 1) (15, 1)
    print(y_train.shape, y_test.shape) #(35, 1) (15, 1)

    w = tf.Variable(tf.random_uniform([3, 1]))  # 난수, 2차원[1, 3] 3열로 만들어줌
    b = tf.Variable(tf.random_uniform([1]))  # 난수, 2차원[1, 3] 3열로 만들어줌

    # 플레이스홀더
    ph_x = tf.placeholder(tf.float32)

    # (50, 1) =  (50, 3) @ (3, 1)
    z = tf.matmul(ph_x, w) + b
    hx = tf.sigmoid(z)
    loss_i = y_train * -tf.log(hx) + (1-y_train) * -tf.log(1-hx) #브로드캐스팅 산술식이라서 y가 리스트이므로 y를 변환
    # loss_i = -y * tf.log(hx) - (1-y) * tf.log(1-hx) #브로드캐스팅 산술식이라서 y가 리스트이므로 y를 변환

    loss = tf.reduce_mean(loss_i)  # cost 계산.. 평균값 3차원->2차원?

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, {ph_x: x_train})
        c = sess.run(loss, {ph_x: x_train})
        if i % 100 == 0:
            print(i, c, sess.run(w), sess.run(b))

        # print(i, c)

    print('-' * 50)

    preds = sess.run(hx, {ph_x: x_test})
    preds = preds.reshape(-1)
    y_test = y_test.reshape(-1)
    print(preds)
    print(y_test)

    preds_bool = np.int32(preds > 0.5)
    print(preds_bool)

    equals = (preds_bool == y_test)
    print(equals)
    print('acc :', np.mean(equals))

    sess.close()






# logistic_regression()
logistic_regression_horese()




