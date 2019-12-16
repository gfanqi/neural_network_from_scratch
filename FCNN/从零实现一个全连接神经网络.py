import numpy as np

from FCNN.losses import Losses
from FCNN.models import Model
from utils.makefigure import show_line_and_data


if __name__ == "__main__":
    # 构建数据
    data1 = np.random.random(size = (100,2)) +2
    data2 = np.random.random(size = (100,2))
    data = np.mat(np.vstack((data1,data2)))

    # 构建分类标签
    y1 = np.ones(100).reshape((100,1))
    y2 = np.zeros(100).reshape((100,1))
    y = np.mat(np.vstack((y1,y2)))

    show_line_and_data(data, y)
    mymodel = Model()
    mymodel.fit(data, y,epochs=1000, lossfunc=Losses.ErrorSqure,learning_rate=0.5)
    test_data = np.random.random(size = (10000,2))*3
    show_line_and_data(test_data,mymodel.predict(test_data))
    print(mymodel.predict(data))



