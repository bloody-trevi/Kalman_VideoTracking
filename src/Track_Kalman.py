import numpy as np


def track_Kalman(xm, ym, first_run):
    """
    영상 처리로 얻은 x, y 값으로 칼만 필터를 거쳐 추정한 x, y값을 구한다.

    :param xm: 영상 처리로 얻은 위치의 x 좌표
    :param ym: 영상 처리로 얻은 위치의 y 좌표
    :param first_run: 초기 실행 여부
    :return: 추정 위치 (x, y)
    """
    global A, H, Q, R, x, P
    if first_run is False:
        dt = 1
        A = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        Q = np.eye(4, k=1, dtype=float)
        R = np.array([[50, 0], [0, 50]])
        x = np.array([[0], [0], [0], [0]])
        P = 100 * np.eye(4, k=1, dtype=int)
        first_run = True

    x_p = A * x
    P_p = A * P * np.transpose(A) + Q

    K = P_p * np.transpose(H) * np.linalg.inv(H * P_p * np.transpose(H) + R)
    z = np.array([[xm], [ym]])
    x = x_p + K * (z - H * x_p)
    P = P_p - K * H * P_p

    xh = x[1]
    yh = x[3]
    return xh, yh
