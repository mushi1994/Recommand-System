#使用cart模型对mnist数据集进行分类
from sklearn import tree
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
#导入数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels
#查看数据
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
print(x_train[0])     
print(y_train[0])
#查看前20张数据具体图片
fig, ax = plt.subplots(nrows=4,ncols=5,sharex='all',sharey='all')
ax = ax.flatten()
for i in range(20):
    img = x_train[i].reshape(28, 28)
    ax[i].imshow(img,cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
#查看数据集中各数字的分布是否均匀
for i in np.arange(0,10):
    print(np.sum(y_train==i))
#构建决策树模型，利用grid_search寻找最优参数
clf = tree.DecisionTreeClassifier(criterion='gini',min_samples_split=4)
parameters={
            'max_depth':[5,10,15,20],
            'min_samples_leaf':[3,4,5,6]
            }
grid_search=GridSearchCV(clf,parameters,cv=3)
grid_search.fit(x_train,y_train) 
#查看模型最佳参数
print(grid_search.best_estimator_) 
print(grid_search.best_params_) 
print(grid_search.best_score_ )
#训练模型并对测试数据进行拟合
clf = tree.DecisionTreeClassifier(criterion='gini',max_depth=15,min_samples_leaf=4,min_samples_split=4)
clf.fit(x_train, y_train)
y_predict= clf.predict(x_test)
print('CART准确率: %0.4lf' % accuracy_score(y_predict,y_test))