# 任务

1.TF-IDF原理。

2.文本矩阵化，使用词袋模型，以TF-IDF特征值为权重。（可以使用Python中TfidfTransformer库）

3.互信息的原理。

4.使用第二步生成的特征矩阵，利用互信息进行特征筛选。

参考资料

[文本挖掘预处理之TF-IDF：文本挖掘预处理之TF-IDF - 刘建平Pinard - 博客园] https://www.cnblogs.com/pinard/p/6693230.html 

[使用不同的方法计算TF-IDF值：使用不同的方法计算TF-IDF值 - 简书] https://www.jianshu.com/p/f3b92124cd2b

[sklearn-点互信息和互信息：sklearn：点互信息和互信息 - 专注计算机体系结构 - CSDN博客] https://blog.csdn.net/u013710265/article/details/72848755

[如何进行特征选择（理论篇）机器学习你会遇到的“坑”：如何进行特征选择（理论篇）机器学习你会遇到的“坑”] https://baijiahao.baidu.com/s?id=1604074325918456186&wfr=spider&for=pc

# 1 TF-IDF原理
TF-IDF(Term Frequency-Inverse Document Frequency, 词频-逆文件频率)，是一种用于资讯检索与资讯探勘的常用加权技术。TF-IDF是一种统计方法，用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。一个词语在一篇文章中出现次数越多, 同时在所有文档中出现次数越少, 越能够代表该文章.。

## 1.1 词频 (term frequency, TF)
TF指的是某一个给定的词语在该文件中出现的次数。这个数字通常会被归一化(一般是词频除以文章总词数), 以防止它偏向长的文件。（同一个词语在长文件里可能会比短文件有更高的词频，而不管该词语重要与否。）

【注意】： 一些通用的词语对于主题并没有太大的作用, 反倒是一些出现频率较少的词才能够表达文章的主题, 所以单纯使用是TF不合适的。权重的设计必须满足：一个词预测主题的能力越强，权重越大，反之，权重越小。所有统计的文章中，一些词只是在其中很少几篇文章中出现，那么这样的词对文章的主题的作用很大，这些词的权重应该设计的较大。IDF就是在完成这样的工作.

公式：
![image](https://github.com/wang-tingxuan/NLP-learning-advanced-punching/blob/master/task-03-%E7%89%B9%E5%BE%81%E9%80%89%E6%8B%A9/TFw.JPG)

## 1.2 逆向文件频率 (inverse document frequency, IDF)
IDF的主要思想是：如果包含词条t的文档越少, IDF越大，则说明词条具有很好的类别区分能力。某一特定词语的IDF，可以由总文件数目除以包含该词语之文件的数目，再将得到的商取对数得到。

公式:
![image](https://github.com/wang-tingxuan/NLP-learning-advanced-punching/blob/master/task-03-%E7%89%B9%E5%BE%81%E9%80%89%E6%8B%A9/IDF.JPG)
分母之所以要加1，是为了避免分母为0

## 1.3 TF-IDF
某一特定文件内的高词语频率，以及该词语在整个文件集合中的低文件频率，可以产生出高权重的TF-IDF。因此，TF-IDF倾向于过滤掉常见的词语，保留重要的词语。
![image](https://github.com/wang-tingxuan/NLP-learning-advanced-punching/blob/master/task-03-%E7%89%B9%E5%BE%81%E9%80%89%E6%8B%A9/TF-IDF.JPG)

# 2 文本矩阵化，使用词袋模型，以TF-IDF特征值为权重。
## 2.1 使用TfidfTransformer
除了考量某词汇在文本出现的频率，还关注包含这个词汇的所有文本的数量，能够削减高频没有意义的词汇出现带来的影响, 挖掘更有意义的特征，相比之下，文本条目越多，Tfid的效果会越显著

# 2.2 使用CountVectorizer
只考虑词汇在文本中出现的频率

# 3 互信息的原理及API
衡量的是两个随机变量之间的相关性，即一个随机变量中包含的关于另一个随机变量的信息量。所谓的随机变量，即随机试验结果的量的表示，可以简单理解为按照一个概率分布进行取值的变量，比如随机抽查的一个人的身高就是一个随机变量。

```python
from sklearn import metrics as mr
mr.mutual_info_score(label,x)
```
# 4 使用第二步生成的特征矩阵，利用互信息进行特征筛选
```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import *
from sklearn.feature_selection import mutual_info_classif

# 读取文件
path = r'E:\jupyter_file\源码\nlp基础\cnews'
train_data = pd.read_csv(path + '\cnews.train.txt', names=['title', 'content'], sep='\t',engine='python',encoding='UTF-8')  # (50000, 2)
test_data = pd.read_csv(path + '\cnews.test.txt', names=['title', 'content'], sep='\t',engine='python',encoding='UTF-8')  # (10000, 2)
val_data = pd.read_csv(path + '\cnews.val.txt', names=['title', 'content'], sep='\t',engine='python',encoding='UTF-8')  # (5000, 2)
x_train = train_data['content']
x_test = test_data['content']
x_val = val_data['content']

y_train  = train_data['title']
y_test = test_data['title']
y_val  = val_data['title']
'''
数据向量化【CountVectorizer】
'''
## 默认配置不去除停用词
count_vec = CountVectorizer()
x_count_train = count_vec.fit_transform(x_train )
x_count_test = count_vec.transform(x_test )

## 去除停用词
count_stop_vec = CountVectorizer(analyzer='word', stop_words='english')
x_count_stop_train = count_stop_vec.fit_transform(x_train)
x_count_stop_test = count_stop_vec.transform(x_test)

## 使用互信息特征筛选
mutual_values = mutual_info_classif(x_count_stop_train , y_train)
print('mean of mutual score', np.median(mutual_values))
print('numbers of mutual score euqal zero ',sum(mutual_values == 0))
## 抽取互信息大于0的idx
idx = [i for i, value in enumerate(mutual_values) if value > 0]
x_train= x_trian[idx]



## 模型训练
mnb_count = MultinomialNB()
mnb_count.fit(x_count_train, y_train)
mnb_count_y_predict = mnb_count.predict(x_count_test)
mnb_count.score(x_count_test, y_test)

mnb_count_stop = MultinomialNB()
mnb_count_stop.fit(x_count_stop_train, y_train)   # 学习
mnb_count_stop_y_predict = mnb_count_stop.predict(x_count_stop_test)
mnb_count_stop.score(x_count_stop_test, y_test)


'''
数据向量化【TfidVectorizer】
'''
## 默认配置不去除停用词
tfid_vec = TfidfVectorizer()
x_tfid_train = tfid_vec.fit_transform(x_train)
x_tfid_test = tfid_vec.transform(x_test)
## 去除停用词
tfid_stop_vec = TfidfVectorizer(analyzer='word', stop_words='english')
x_tfid_stop_train = tfid_stop_vec.fit_transform(x_train)
x_tfid_stop_test = tfid_stop_vec.transform(x_test)
## 模型训练
mnb_tfid = MultinomialNB()
mnb_tfid.fit(x_tfid_train, y_train)
mnb_tfid_y_predict = mnb_tfid.predict(x_tfid_test)
mnb_tfid.score(x_tfid_test, y_test)## 


mnb_tfid_stop = MultinomialNB()
mnb_tfid_stop.fit(x_tfid_stop_train, y_train)   # 学习
mnb_tfid_stop_y_predict = mnb_tfid_stop.predict(x_tfid_stop_test)    # 预测
mnb_tfid_stop.score(x_count_stop_test, y_test)
```
