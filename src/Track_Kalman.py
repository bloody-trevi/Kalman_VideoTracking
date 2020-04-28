import numpy as np
from src.Kalman_Params_class import KalmanParam


def track_Kalman(xm, ym, param):

    """
    영상 처리로 얻은 x, y 값으로 칼만 필터를 거쳐 추정한 x, y값을 구한다.

    :param xm: 영상 처리로 얻은 위치의 x 좌표
    :param ym: 영상 처리로 얻은 위치의 y 좌표
    :return: 추정 위치 (x, y)
    """

    A, H, Q, R = param.get_mats()
    x = param.get_x()
    P = param.get_P()

    x_p = np.dot(A, x)
    P_p = A * P * np.transpose(A) + Q

    K_1 = np.dot(P_p, np.transpose(H))
    K_2 = np.linalg.inv(H.dot(np.dot(P_p, np.transpose(H))) + R) # inv(H*P_p*H' + R)
    K = np.dot(K_1, K_2)

    z = np.transpose(np.array([xm, ym]))
    x_ = z - np.dot(H, x_p)
    x = x_p + np.dot(K, x_)
    P = P_p - K.dot(np.dot(H, P_p))

    xh = x[0]
    yh = x[2]
    param.set_x_P(x, P)
    return xh, yh, param
