import numpy as np
import pandas as pd
import xlrd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib as mpl  
import matplotlib.pyplot as plt 

from MaxEnt import MaxEnt

# 划分测试数据和训练数据
def validate(X_data, y_data, ratio=0.15):
    N = X_data.shape[0] # 数据总数
    size = int(N * ratio) 
    inds = np.random.permutation(range(N)) # 随机排列数据下标
    for i in range(int(N / size)):
        test_ind = inds[i * size:(i + 1) * size]
        train_ind = list(set(range(N)) - set(test_ind))
        yield X_data[train_ind], y_data[train_ind], X_data[test_ind], y_data[
            test_ind]


# 对数据进行归一化处理
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


# 对数据进行标准化处理
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

# 绘制ROC曲线
def plot_roc(labels, predict_prob):
    false_positive_rate,true_positive_rate,thresholds=roc_curve(labels, predict_prob)###计算真正率和假正率
    roc_auc=auc(false_positive_rate, true_positive_rate)###计算auc的值
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate,'b',label='AUC = %0.4f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()

# 往Excel中写入数据
def write_data_excel():
    frame = pd.DataFrame([],
                     index=['exp1','exp2','exp3','exp4'],
                     columns=['jan2015','Fab2015','Mar2015','Apr2005'])
    print(frame)
    frame.to_excel("data.xlsx")

# 主方法
if __name__ == '__main__':

    #1、从Excel文件中读取出数据
    data_excel = xlrd.open_workbook("实验数据3.xlsx")
    table = data_excel.sheets()[0] # 通过索引顺序获取
    #table = pd.read_excel("实验数据3.xlsx", sheet_name = 0)
    
    # nrows = table.nrows
    nrows = table.nrows
    ncols = table.ncols
    x_data = []
    i = 1
    while i < nrows:
         x_data.append(table.row_values(i))
         i += 1
    x_data = np.array(x_data)
    x_data = np.delete(x_data,0,axis=1)
    y_data = x_data[:, 0]
    x_data = np.delete(x_data,0,axis=1)
    
    #2、对经纬度数据进行归一化处理  
    x_data[:, [0, 1]] = normalization(np.array(x_data[:, [0, 1]]).astype(float))

    #3、对海拔降雨量等环境因素进行标准化处理
    x_data[:, [2]] = standardization(np.array(x_data[:, [2]]).astype(float))

    test_data = x_data[:15]
    x_data = x_data[16:]
    test_labels = y_data[:15]
    y_data = y_data[16:]

    print(test_labels)

    g = validate(x_data, y_data, ratio=0.3)

    #划分训练集测试集，取9.45%的测试集（刚好等于209）。

    # ME = MaxEnt(maxstep=3)
    # ME.fit(train_data, train_labels)
    for item in g:
        X_train, y_train, X_test, y_test = item
        ME = MaxEnt(maxstep=3)
        ME.fit(X_train, y_train)
        score = 0
        single_label = [] # 保存每次测试结果的分类
        single_score = [] # 保存每次测试结果的分类概率
        for X, y in zip(test_data, test_labels): 
            single = ME.predict(X)
            single_label.append(int(single['label'][0: 1]))
            single_score.append(single['score'])
            if single['label'] == y:
                score += 1
    
        # single_label = np.array(single_label) - np.array([1]) # 分类结果改成只有0、1

        #fpr,tpr,threshold = roc_curve(single_label, np.array(single_score)) ###计算真正率和假正率
        #roc_auc = auc(fpr,tpr) ###计算auc的值
        #print(y_test)
        print(single_label)
        
        
        plot_roc(single_label, np.array(single_score))


        print(score / len(test_labels))
    
