from FCNN.layers import Layers, Dense
from FCNN.models import Model


class Mymodel(Model):
	def __init__(self):
		'''
		初始化层
		'''
		super().__init__()
		self.layer1 = Dense(2)
		self.layer2 = Dense(2)
		self.layer3 = Dense(3)
		self.layer4 = Dense(1)

	def call(self, data):
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


	def backproandrenew(self, lossfunc=None, learning_rate=None):
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
		return last_grad