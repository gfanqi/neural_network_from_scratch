import numpy as np

from FCNN.losses import SqureLossError
from FCNN.mymodel import Mymodel
from utils.makefigure import show_line_and_data

if __name__ == "__main__":
    # 构建数据
    data1 = np.random.random(size=(100, 1)) + 2
    data2 = np.random.random(size=(100, 1))
    data = np.mat(np.vstack((data1, data2)))

    # 构建分类标签
    y1 = np.ones(100).reshape((100, 1))
    y2 = np.zeros(100).reshape((100, 1))
    y = np.mat(np.vstack((y1, y2)))

    show_line_and_data(data, y)
    mymodel = Mymodel()
    mymodel.fit(data, y, epochs=1000, lossfunc=SqureLossError, learning_rate=0.05)
    # test_data = np.random.random(size=(1000, 2)) * 3
    test_data = data
    show_line_and_data(test_data, mymodel.predict(test_data))

