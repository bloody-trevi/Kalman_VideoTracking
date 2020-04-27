"""
    Main loop
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
from src.Track_Kalman import track_Kalman
import numpy as np
from matplotlib import pyplot as plt
from src.Kalman_Params_class import KalmanParam
import random


if __name__ == '__main__':
    data = np.load("data.npy")
    # print(data)

    num_locations = data.size

    xm_saved = np.zeros((num_locations, 2))
    xh_saved = np.zeros((num_locations, 2))

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
    x = np.transpose(np.array([0, 0, 0, 0]))
    P = 100 * np.eye(4, k=0, dtype=int)
    param = KalmanParam(A, H, Q, R, x, P)

    for k in range(1, 307, 10): # 1, num_locations
        rand1 = random.randrange(-50, 50)
        rand2 = random.randrange(-50, 50)
        xm, ym = (int(data[k][0]) + rand1, int(data[k][1]) + rand2)
        xh, yh, param = track_Kalman(xm, ym, param)

        ax1.plot(xm, ym, 'r.')
        ax1.plot(xh, yh, 'b.')

        xm_saved[k - 1, :] = np.array([xm, ym])
        xh_saved[k - 1, :] = np.array([xh, yh])

    ax2 = fig.add_subplot(221)
    ax2.plot(xm_saved[:, 0], xm_saved[:, 1], 'r.')
    ax3 = fig.add_subplot(222)
    ax3.plot(xh_saved[:, 0], xh_saved[:, 1], 'b.')
    plt.show()
