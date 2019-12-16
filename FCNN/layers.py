import numpy as np

class Layers:
	class Dense:
		'''
		全连接层
		'''

		def __init__(self, output_category):
			'''
			接收并初始化一个输出维度，用于确定这一层w的维度，以及用于梯度计算
			:param output_category:
			'''
			self.output_category = output_category

		def __call__(self, Input):
			'''
			使用魔法方法，实例化对象后，随机的方式初始化w参数，
			实例化输入数据，计算本层前向传播方式
			:param Input:
			:return:
			'''
			w_shape = (Input.shape[1], self.output_category)
			self.w = np.mat(np.random.random(size=w_shape))
			self.Input = Input
			result = self.Input * self.w
			self.result = result
			return result

		def backpropagation(self, w_grad_from_next_layer=None, learning_rate=None):
			'''
			反向传播算法的数学描述，公式参考
			https://zhuanlan.zhihu.com/p/24863977

			:param w_grad_from_next_layer:从下一层传过来的梯度
			:param learning_rate:学习率
			:return:
			'''
			mid_w_grad = np.mat(np.kron(np.eye(self.output_category), self.Input))
			self.w_grad = w_grad_from_next_layer * mid_w_grad
			mid_x_grad = np.kron(self.w.T, np.eye(self.Input.shape[0]))
			self.x_grad = w_grad_from_next_layer * mid_x_grad
			if learning_rate is not None:
				self.w = self.w - learning_rate * self.w_grad.T
			return self.x_grad

	class Relu:
		'''
		Relu激活层
		'''

	class Sigmod:
		'''
		Sigmod激活层
		'''