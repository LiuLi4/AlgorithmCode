import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
import xlrd

def multiple_linear_regression(x, y):

    #划分训练集测试集，取9.45%的测试集（刚好等于209）。
    train_data,test_data, train_labels, test_labels = train_test_split(x, y, test_size = 0.0945, random_state = 0)
    print(train_data)#看一下测试集长度

    #导入线性回归模型 拟合模型
    model=LinearRegression()
    model.fit(train_data, train_labels)
    print('模型系数是：',model.coef_)

    #使用模型在测试集和训练集上预测。
    y_train_predict=model.predict(train_data)
    y_train_predict=y_train_predict 
    y_test_predict=model.predict(test_data)
    y_test_predict=y_test_predict

    print('训练集上的MAE和RMSE:')
    print(mean_absolute_error(y_train_predict, train_labels))
    print(math.sqrt(mean_squared_error(y_train_predict, train_labels)))
    print('测试集上的MAE和RMSE:')
    print(mean_absolute_error(y_test_predict, test_labels))
    print(math.sqrt(mean_squared_error(y_test_predict, test_labels) ))

    #在测试集上的预测效果
    plt.scatter(test_labels, y_test_predict)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    _ = plt.plot([-1, 0], [-1, 0])


    #在训练集上的预测效果
    # plt.plot(test_labels.values[0:1])
    plt.plot(test_labels)
    plt.plot(y_test_predict[0:1])
    plt.legend(('y_test','predict_y'))
    plt.title('test')
    plt.show()

# 对数据进行归一化处理
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


# 对数据进行标准化处理
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

# 主方法
if __name__ == '__main__':

    #1、从Excel文件中读取出数据
    data_excel = xlrd.open_workbook("data.xlsx")
    table = data_excel.sheets()[0] # 通过索引顺序获取
    
    # nrows = table.nrows
    nrows = table.nrows
    ncols = table.ncols
    x_data = []
    i = 1
    while i < nrows:
         x_data.append(table.row_values(i))
         i += 1
    x_data = np.array(x_data)
    y_data = x_data[:, 0]
    y_data = np.delete(y_data,15,axis=0)

    x_data = np.delete(x_data,0,axis=1)
    x_data = np.delete(x_data,15,axis=0)

    multiple_linear_regression(x_data, y_data)
