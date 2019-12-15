import numpy as np
import matplotlib.pyplot as plt


def show_line_and_data(data, value, w=None, is_show=True):
    data = np.array(data)
    value = np.array(value)
    value = value.flatten()

    # w_shape = w.shape
    data_shape = data.shape
    value_shape = value.shape
    if w is not None:
        slope = np.array(w)[0][0] / np.array(w)[1][0]
        inx = np.array(w)[2][0] / np.array(w)[1][0]
        # print(slope,inx)
        prex = np.array(range(-10, 20))
        prey = -slope * prex - inx
        plt.plot(prex, prey)
    plt.scatter(data[:, 0], data[:, 1],c=value)
    # plt.ylim(-10, 10)
    plt.show()


def show_loss(losses):
    x = list(range(len(losses)))
    plt.plot(x, losses)
    plt.show()
