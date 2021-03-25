import numpy as np

class Layers:
	def __call__(self, *args, **kwargs):
		return self.call(*args, **kwargs)

	def call(self,*args,**kwargs):
		raise ImportError

	def backpropagation(self, w_grad_from_next_layer=None, learning_rate=None, ):
		raise ImportError
	pass

class Dense(Layers):
	'''
	全连接层
	'''

	def __init__(self, output_category):
		'''
		接收并初始化一个输出维度，用于确定这一层w的维度，以及用于梯度计算
		:param output_category:
		'''
		self.output_category = output_category

	def call(self, Input):
		'''
		使用魔法方法，实例化对象后，随机的方式初始化w参数，
		实例化输入数据，计算本层前向传播方式
		:param Input:
		:return:
		'''
		w_shape = (Input.shape[1], self.output_category)
		b_shape = (Input.shape[0], self.output_category)
		self.w = np.mat(np.random.random(size=w_shape))
		self.b = np.mat(np.random.random(size=b_shape))
		self.Input = Input
		result = self.Input * self.w +self.b
		self.result = result
		return result

	def backpropagation(self, w_grad_from_next_layer=None, learning_rate=None, use_bias=False):
		'''
		反向传播算法的数学描述，公式参考
		https://zhuanlan.zhihu.com/p/24863977
		公式 Y = X * w +b ，
		dY = dX * w + X * dw + db
		   = I * dX * w + X * dw * I + I * db * I   (I是单位矩阵,
		   公式里每个I都不一样维度，具体是多少要参考它与谁相乘）
		vec(dY) = np.kron(w,I)*vec(dX)  + np.kron(I,w)*vec(dw) + np.kron(I,I)*vec(db)
		:param w_grad_from_next_layer:从下一层传过来的梯度
		:param learning_rate:学习率
		:return:
		'''
		mid_w_grad = np.mat(np.kron(np.eye(self.output_category), self.Input))
		self.w_grad = w_grad_from_next_layer * mid_w_grad


		mid_x_grad = np.kron(self.w.T, np.eye(self.Input.shape[0]))
		self.x_grad = w_grad_from_next_layer * mid_x_grad

		if use_bias == True:
			mid_b_grad = np.kron(np.eye(self.output_category), np.eye(self.Input.shape[0]))
			self.b_grad = w_grad_from_next_layer * mid_b_grad

		if learning_rate is not None:
			self.w = self.w - learning_rate * self.w_grad.T
			if use_bias == True:
				self.b = self.b - learning_rate * self.b_grad
		return self.x_grad

class Activation(Layers):
	'''
	激活层
	'''
	def __init__(self, activate_func):
		'''
		传入激活方程类
		:param activate_func:
		'''
		self.activate_func = activate_func

	def call(self, Input):
		'''
		使用魔法方法，实例化对象后，随机的方式初始化w参数，
		实例化输入数据，计算本层前向传播方式
		:param Input:
		:return:
		'''
		self.Input = Input
		self.activate_func_obj = self.activate_func(self.Input)
		result = self.activate_func_obj.call()
		self.result = result
		return result

	def backpropagation(self, w_grad_from_next_layer=None, learning_rate=None, ):
		'''
		反向传播算法的数学描述，公式参考
		https://zhuanlan.zhihu.com/p/24863977
		公式 Y = f(X)   (f是逐元素函数)
			dY = df(X)
			   = f'(X) ⊙ dX (⊙表示出逐元素相乘，也就是通缩意义上的对应位置相乘）
			所以
			vec(dY) = np.diagflat(f'(X))* vec(dX)

		:param w_grad_from_next_layer:从下一层传过来的梯度
		:param learning_rate:学习率
		:return:
		'''

		mid_x_grad = self.activate_func_obj.grad()
		self.x_grad = w_grad_from_next_layer * mid_x_grad
		return self.x_grad







