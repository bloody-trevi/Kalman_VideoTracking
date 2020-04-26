"""
    Main loop
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
from src.Track_Kalman import track_Kalman
import numpy as np
from matplotlib import pyplot as plt
from src.Kalman_Params_class import KalmanParam


if __name__ == '__main__':
    data = np.load("data.npy")
    # print(data)

    num_locations = data.size

    xm_saved = np.zeros((2, 24))
    xh_saved = np.zeros((2, 24))

    fig = plt.figure()
    ax1 = fig.add_subplot(212)

    # 칼만 필터 파라미터 초기화
    # 물체가 등속 운동을 한다고 가정하고 시스템 모델 설정

    # 파라미터 갱신할 클래스
    dt = 1
    A = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    Q = np.eye(4, k=0, dtype=float)
    R = np.array([[50, 0], [0, 50]])
    x = np.array([[0], [0], [0], [0]])
    P = 100 * np.eye(4, k=0, dtype=int)
    param = KalmanParam(A, H, Q, R, x, P)

    for k in range(1, num_locations):
        #
        xm, ym = (data[k][0], data[k][1])
        xh, yh, param = track_Kalman(xm, ym, param)

        ax1.plot(xm, ym, 'r*', label="Measured")
        ax1.plot(xh, yh, 'bs', label="Filtered")
        ax1.legend(loc="best")

        xm_saved[:, k] = np.array([[xm], [ym]])
        xh_saved[:, k] = np.array([[xh], [yh]])

    ax2 = fig.add_subplot(221)
    ax2.plot(xm_saved[1, :], xm_saved[2, :], '*')
    ax3 = fig.add_subplot(222)
    ax3.plot(xh_saved[1, :], xh_saved[2, :], '$')
    plt.show()
