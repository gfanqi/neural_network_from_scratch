from FCNN.layers import Layers

class Model:
    def __init__(self):
        '''
        初始化
        '''

    def call(self,data):
        '''
        前向传播逻辑
        :return:
        '''

    def fit(self, data, y, lossfunc,learning_rate=1,epochs=1):
        '''
        训练参数
        :param data:
        :param y:
        :param lossfunc:
        :param learning_rate:
        :param epochs:
        :return:
        '''
        self.y = y
        self.X = data
        for epoch in range(epochs):
            self.call(data)
            self.backproandrenew(learning_rate=learning_rate,lossfunc=lossfunc)

    def backproandrenew(self,lossfunc=None, learning_rate=None):
        '''
        反向传播算法，更新参数
        :param lossfunc:
        :param learning_rate:
        :return:
        '''

    def predict(self, data):
        '''
        预测数据
        :param data:
        :return:
        '''
        return self.call(data)







