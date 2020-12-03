import numpy as np
import matplotlib.pyplot as plt
import matplotlib

print(np.__version__)

def show_line_and_data(data, value, w=None, is_show=True):
	# 这个函数是用来作图的
	data = np.array(data)
	value = np.array(value)
	value = value.flatten()

	if w is not None:
		# w是分割超平面的参数，
		slope = np.array(w)[0][0] / np.array(w)[1][0]
		inx = np.array(w)[2][0] / np.array(w)[1][0]
		# print(slope,inx)
		prex = np.array(range(-10, 20))
		prey = -slope * prex - inx
		plt.plot(prex, prey)
	plt.scatter(data[:, 0], data[:, 1], c=value)
	# plt.ylim(-10, 10)
	plt.show()


if __name__ == "__main__":
	# 构建数据
	data1 = np.random.random(size=(100, 2)) + 2
	data2 = np.random.random(size=(100, 2))
	data = np.mat(np.vstack((data1, data2)))

	# 构建分类标签
	y1 = np.ones(100).reshape((100, 1))
	y2 = np.zeros(100).reshape((100, 1))
	y = np.mat(np.vstack((y1, y2)))

	show_line_and_data(data, y)
	# mymodel = Mymodel()
	# mymodel.fit(data, y, epochs=100, lossfunc=Losses.ErrorSqure, learning_rate=0.5)
	# test_data = np.random.random(size=(10000, 2)) * 3
	# show_line_and_data(test_data, mymodel.predict(test_data))
	# print(mymodel.predict(data))
