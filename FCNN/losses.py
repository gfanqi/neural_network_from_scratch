import numpy as np

class Losses:
    '''
    损失函数类
    '''
    def __init__(self):
        pass

    class ErrorSqure:
        '''
        平方误差类
        '''
        def __init__(self,y,pred):
            self.y=y
            self.pred = pred

        def loss(self):
            '''
            计算损失，并返回
            :return:
            '''
            result = np.linalg.norm(self.y-self.pred)
            return result

        def grad(self):
            '''
            梯度计算
            :return:
            '''
            result = np.mat(2*(self.pred-self.y).T)
            return result
