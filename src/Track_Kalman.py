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

    x_p = A * x
    P_p = A * P * np.transpose(A) + Q

    K = P_p * np.transpose(H) * np.linalg.inv(H * P_p * np.transpose(H) + R)
    z = np.array([[xm], [ym]])
    x = x_p + K * (z - H * x_p)
    P = P_p - K * H * P_p

    xh = x[1]
    yh = x[3]
    param.set_x_P(x, P)
    return xh, yh, param
