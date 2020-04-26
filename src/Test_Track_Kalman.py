"""
    Main loop
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
from src.Track_Kalman import track_Kalman
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    num_img = 24

    xm_saved = np.zeros((2, 24))
    xh_saved = np.zeros((2, 24))

    fig = plt.figure()
    ax1 = fig.add_subplot(212)
    for k in range(1, num_img):
        #
        xm, ym = get_ball_pos(k)
        xh, yh = track_Kalman(xm, ym)

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
