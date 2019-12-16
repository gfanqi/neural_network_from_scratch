from FCNN.layers import Layers


class Model:
    def __init__(self):
        '''
        初始化层
        '''
        self.layer1 = Layers.Dense(2)
        self.layer2 = Layers.Dense(2)
        self.layer3 = Layers.Dense(3)
        self.layer4 = Layers.Dense(1)

    def call(self,data):
        '''
        前向传播逻辑
        :param data:
        :return:
        '''
        x = self.layer1(data)
        x = self.layer2(x)
        x = self.layer3(x)
        output = self.layer4(x)
        return output

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

    def predict(self,data):
        '''
        预测数据
        :param data:
        :return:
        '''
        return self.call(data)


    def backproandrenew(self,lossfunc=None,learning_rate=None):
        '''
        反向传播算法，更新参数
        :param lossfunc:
        :param learning_rate:
        :return:
        '''
        mylossfunc = lossfunc(self.y, self.layer4.result)
        last_grad = mylossfunc.grad()
        last_grad = self.layer4.backpropagation(last_grad)
        last_grad = self.layer3.backpropagation(last_grad)
        last_grad = self.layer2.backpropagation(last_grad)
        last_grad = self.layer1.backpropagation(last_grad)