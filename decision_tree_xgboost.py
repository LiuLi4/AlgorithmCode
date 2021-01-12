import xgboost as xgb
import numpy as np
# 自己实现loss function,softmax函数
def log_reg(y_hat, y):
	p = 1.0 / (1.0 + np.exp(-y_hat))
	g = p - y.get_label()
	h = p * (1.0 - p)
	return g, h
# 自己实现错误率计算
def error_rate(y_hat, y):
	return 'error', float(sum(y.get_label() != (y_hat > 0.5))) / len(y_hat)
 
if __name__ == '__main__':
	# 读取数据
	data_train = xgb.DMatrix('agaricus_train.txt')
	data_test = xgb.DMatrix('agaricus_test.txt')
	print('data_train:\n', data_train)
	print(type(data_train))
 
	# 设定相关参数
	param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
	# param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'reg:logistic'}
	watchlist = [(data_test, 'eval'), (data_train, 'train')]
	n_round = 10
	bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist, obj=log_reg, feval=error_rate)
 
	# 计算错误率
	y_hat = bst.predict(data_test)
	y = data_test.get_label()
	print('y_hat:\n', y_hat)
	print('y:\n', y)
 
	error = sum(y != (y_hat > 0.5))
	error_rate = float(error) / len(y_hat)
 
	print('total samples:%d' % len(y_hat))
	print('the wrong numbers:%d' % error)
	print('error ratio:%.3f%%' % error_rate)