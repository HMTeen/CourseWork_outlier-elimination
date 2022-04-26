import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

def PLOT_hist(Data1, Data2, Data3, Data4, sub_figure_title, sub_figure_skewness):
    ############ 绘图 ############
    left  = 0.125  # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.9      # the top of the subplots of the figure
    wspace = 0.35   # the amount of width reserved for blank space between subplots,
                # expressed as a fraction of the average axis width
    hspace = 0.45   # the amount of height reserved for white space between subplots,
                # expressed as a fraction of the average axis height

    # 仅原始数据
    fig = plt.figure(figsize=(8,6))
    plt.subplots_adjust(left, bottom, right, top, wspace, hspace)

    f1 = fig.add_subplot(221)
    f1.hist(Data1, 40)
    f1.set_title(sub_figure_title[0] + '  ||  skew = {:.4}'.format(sub_figure_skewness[0]))  

    f2 = fig.add_subplot(222)
    f2.hist(Data2, 40)
    f2.set_title(sub_figure_title[1] + '  ||  skew = {:.4}'.format(sub_figure_skewness[1]))  

    f3 = fig.add_subplot(223)
    f3.hist(Data3, 40)
    f3.set_title(sub_figure_title[2] + '  ||  skew = {:.4}'.format(sub_figure_skewness[2]))  

    f4 = fig.add_subplot(224)
    f4.hist(Data4, 40)
    f4.set_title(sub_figure_title[3] + '  ||  skew = {:.4}'.format(sub_figure_skewness[3]))  
 
    plt.show()


def outlier_elimination(data):
    df = pd.Series(data)
    mean = df.mean()
    std = df.std()
    temp_index = []
    for key, value in df.items():
        tmp = np.abs(value - mean) > 3*std
        if tmp:
            temp_index.append(key)
    df.drop(temp_index, inplace=True)
    return df.to_numpy()

shape, scale = 1., 3.
Initial_Data = np.random.gamma(shape, scale, 2000) / 20 + 0.001    # 生成2000个随机数，并进行区间缩放微调。

Initial_Data_sqrt = np.sqrt(Initial_Data)
Initial_Data_sqrt4 = np.power(Initial_Data, 1/4)
Initial_Data_sqrt5 = np.power(Initial_Data, 1/5)
Initial_Data_sqrt6 = np.power(Initial_Data, 1/6)

temp_title = ['Initial_Data', 's_sqrt', 's_sqrt4', 's_sqrt6']
temp_skewness = [stats.skew(Initial_Data), stats.skew(Initial_Data_sqrt), stats.skew(Initial_Data_sqrt4), stats.skew(Initial_Data_sqrt6)]
PLOT_hist(Initial_Data, Initial_Data_sqrt, Initial_Data_sqrt4, Initial_Data_sqrt6, temp_title, temp_skewness)

temp_data = [Initial_Data, Initial_Data_sqrt, Initial_Data_sqrt4, Initial_Data_sqrt6]

s_new = outlier_elimination( Initial_Data_sqrt4 )
print( stats.skew(Initial_Data_sqrt4), stats.skew(s_new))

temp_title = ['s_sqrt4', 's_sqrt4_outlier', 's_sqrt4', 's_sqrt4_outlier']
temp_skewness = [stats.skew(Initial_Data_sqrt4), stats.skew(s_new),stats.skew(Initial_Data_sqrt4), stats.skew(s_new)]
PLOT_hist(Initial_Data_sqrt4, s_new, Initial_Data_sqrt4, s_new, temp_title, temp_skewness)

