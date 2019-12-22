import numpy as np

class Funcs:
    '''
    方程
    '''
    class Relu:
        '''
        Relu函数
        '''

        def __init__(self,Input):
            self.Input = Input
            pass

        def call(self):
            '''
            计算结果，并返回
            :return:
            '''

            result = np.multiply((self.Input>0),self.Input)
            return result

    class Softmax:
        '''
        Relu函数
        '''

        def __init__(self,Input,axis=None):
            self.Input = Input
            self.__axis = axis
            pass

        def call(self):
            '''
            计算结果，并返回
            :return:
            '''
            if self.__axis is None:
                self.__axis = -1
            result = np.exp(self.Input)/np.sum(np.exp(self.Input),axis=self.__axis)
            return result


        def grad(self):
            '''
            梯度计算，
            :return:
            '''
            pass

    class SqureLossError:
        '''
        平方误差类
        '''
        def __init__(self,y,pred):
            self.y=y
            self.pred = pred

        def call(self):
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
