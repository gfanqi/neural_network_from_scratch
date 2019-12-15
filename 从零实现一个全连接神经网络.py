import numpy as np
from utils.makefigure import show_line_and_data

class Layers:
    class Dense:
        '''

        '''
        def __init__(self,output_category):
            self.output_category = output_category


        def __call__(self,Input):
            w_shape = (Input.shape[1],self.output_category)
            self.w = np.mat(np.random.random(size=w_shape))
            self.Input = Input
            result = self.Input * self.w
            self.result = result
            return result


        def backpropagation(self,w_grad_from_next_layer=None,learning_rate=None):
            mid_w_grad = np.mat(np.kron(np.eye(self.output_category) , self.Input))
            self.w_grad = w_grad_from_next_layer*mid_w_grad
            mid_x_grad = np.kron(self.w.T,np.eye(self.Input.shape[0]))
            self.x_grad = w_grad_from_next_layer*mid_x_grad
            if learning_rate is not None:
                self.w = self.w-learning_rate*self.w_grad.T
            return self.x_grad
    class Relu:
        '''
        Relu激活层
        '''

class Model:
    def __init__(self):
        self.layer1 = Layers.Dense(2)
        self.layer2 = Layers.Dense(2)
        self.layer3 = Layers.Dense(3)
        self.layer4 = Layers.Dense(1)

    def call(self,data):
        x = self.layer1(data)
        x = self.layer2(x)
        x = self.layer3(x)
        output = self.layer4(x)
        return output

    def fit(self, data, y, lossfunc,learning_rate,epochs=1,):
        self.y = y
        self.X = data
        for epoch in range(epochs):
            self.call(data)
            self.backproandrenew(learning_rate=learning_rate,lossfunc=lossfunc)

    def predict(self,data):
        return self.call(data)


    def backproandrenew(self,lossfunc=None,learning_rate=None):
        mylossfunc = lossfunc(self.y, self.layer4.result)
        last_grad = mylossfunc.grad()
        last_grad = self.layer4.backpropagation(last_grad)
        last_grad = self.layer3.backpropagation(last_grad)
        last_grad = self.layer2.backpropagation(last_grad)
        last_grad = self.layer1.backpropagation(last_grad)



class Losses:
    def __init__(self):
        pass

    class ErrorSqure:
        def __init__(self,y,pred):
            self.y=y
            self.pred = pred

        def loss(self):
            result = np.linalg.norm(self.y-self.pred)
            return result

        def grad(self):
            result = np.mat(2*(self.pred-self.y).T)
            return result


if __name__ == "__main__":
    data1 = np.random.random(size = (100,2)) +2
    data2 = np.random.random(size = (100,2))
    data = np.mat(np.vstack((data1,data2)))

    y1 = np.ones(100).reshape((100,1))
    y2 = np.zeros(100).reshape((100,1))
    y = np.mat(np.vstack((y1,y2)))

    show_line_and_data(data, y)
    mymodel = Model()
    mymodel.fit(data, y,epochs=1000, lossfunc=Losses.ErrorSqure,learning_rate=0.5)
    test_data = np.random.random(size = (10000,2))*3
    # show_line_and_data()
    show_line_and_data(test_data,mymodel.predict(test_data))
    print(mymodel.predict(data))



