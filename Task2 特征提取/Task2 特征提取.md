# 1. 基本文本处理技能

## 1.1 分词的概念

分词就是将连续的字序列按照一定的规范重新组合成词序列的过程。中文分词就是将一个汉字序列切分成一个一个单独的词。现有的分词方法可分为三大类：基于字符串匹配的分词方法、基于理解的分词方法和基于统计的分词方法。

最大匹配法：最大匹配是指以词典为依据，取词典中最长单词为第一个次取字数量的扫描串，在词典中进行扫描（为提升扫描效率，还可以跟据字数多少设计多个字典，然后根据字数分别从不同字典中进行扫描）。

在字符串匹配的分词方法中，根据扫描方向可以分为正向匹配和逆向匹配，根据不同情况优先匹配的情况，可以分为最大匹配和最小匹配。

正向最大匹配法：从左向右最大长度匹配
逆向最大匹配法：从右向左最大长度匹配
双向最大匹配法：从左向右和从右向左两次扫描最大匹配。正向和逆向两种算法都切一遍，然后根据大颗粒度词越多越好，非词典词和单字词越少越好的原则，选取其中一种分词结果输出。分词目标：将正向最大匹配算法和逆向最大匹配算法进行比较，从而确定正确的分词方法。
双向最大匹配法的算法流程：
比较正向最大匹配和逆向最大匹配结果: 如果分词数量结果不同，那么取分词数量较少的那个; 如果分词数量结果相同 , 若分词结果相同，可以返回任何一个， 分词结果不同，返回字符比较少的那个。

## 1.2 词、字符频率统计

使用Python中的collections.Counter模块：
```pyhton
import collections
words = '我是一名学生，我热爱学习，我还爱看书'
words_counts = collections.Counter(words)  # 在底层中，Counter是一个字典，在元素和它们出现的次数间做了映射
top_three = words_counts.most_common(3)  # 返回出现次数前三的元素
print(top_three)
```
输出:
```
[('我', 3), ('学', 2), ('，', 2)]
```

# 2. 概念
## 2.1 语言模型中unigram、bigram、trigram的概念；
unigram 一元分词，把句子分成一个一个的汉字
bigram 二元分词，把句子从头到尾每两个字组成一个词语
trigram 三元分词，把句子从头到尾每三个字组成一个词语.
## 2.2 unigram、bigram频率统计；（可以使用Python中的collections.Counter模块，也可以自己寻找其他好用的库）
```python
import collections
text = u"""北京时间8月6日，NBA官网邀请专家团评选出过去10年NBA最佳阵容，勒布朗-詹姆斯和凯文-杜兰特领衔过去10年的最佳一阵。

　　注：这并不是NBA官方评选的最佳阵容，而是NBA官网邀请专家团评选出过去10年NBA最佳阵容，从2009-10赛季算起。每套阵容由两名后卫和三名前场球员组成。

　　过去10年的最佳一阵名单：斯蒂芬-库里，詹姆斯-哈登，勒布朗-詹姆斯，凯文-杜兰特，考瓦伊-莱昂纳德。"""
print("-------------counter unigram--------------")
unigram_counter = collections.Counter([text[i] for i in range(0,len(text))])
for k,v in unigram_counter.items():
    print(k,v)

print("-------------counter bigram--------------")
bigram_counter = collections.Counter([(text[i],text[i+1]) for i in range(0,len(text)-1)])
for k,v in bigram_counter.items():
    print(k,v)

print("-------------counter bigram--------------")
bigram_counter = collections.Counter([(text[i],text[i+1],text[i+2]) for i in range(0,len(text)-2)])
for k,v in bigram_counter.items():
    print(k,v)
```

# 3. 文本矩阵化：要求采用词袋模型且是词级别的矩阵化
步骤有：
# 3.1 分词（可采用结巴分词来进行分词操作，其他库也可以）；
```python
import jieba
import re
stopwords = {}
fstop = open('stop_words.txt', 'r',encoding='utf-8',errors='ingnore')
for eachWord in fstop:
    stopwords[eachWord.strip()] = eachWord.strip()  #停用词典
fstop.close()
f1=open('t1.txt','r',encoding='utf-8',errors='ignore')
f2=open('fenci.txt','w',encoding='utf-8')

line=f1.readline()
while line:
    line = line.strip()  #去前后的空格
    line = re.sub(r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+", " ", line) #去标点符号
    seg_list=jieba.cut(line,cut_all=False)  #结巴分词
    outStr=""
    for word in seg_list:
        if word not in stopwords:
            outStr+=word
            outStr+=" "
    f2.write(outStr)
    line=f1.readline()
f1.close()
f2.close()
```
# 3.2 去停用词；构造词表。
```python 
count = 0
dic = {}
for i, j in wordList.items():
    if j > 2:
        dic[i] = count 
        count += 1
#反向词表
resDic = {}
for i, j in dic.items():
    resDic[j] = i
```
# 3.3 每篇文档的向量化。
```python
import jieba
f=open("fenci.txt","r",encoding="utf-8")
t=f.read()
t=set(t.split())
# print(t)
corpus_dict=dict(zip(t,range(len(t))))
print(corpus_dict)
# 建立句子的向量表示
text='''韦斯特一心瞄准自由市场 怎奈伤病挡路 钱途渺茫'''
text1 = list(jieba.cut(text, cut_all=False))
print(text1)
def vector_rep(text, corpus_dict):
    vec,vec2 = [],[]
    for key in corpus_dict.keys():
        if key in text:
            vec.append((corpus_dict[key], text.count(key)))
            vec2.append(text.count(key))
        else:
            vec.append((corpus_dict[key], 0))
            vec2.append(0)
    vec = sorted(vec, key= lambda x: x[0])
    return vec,vec2
vec1,vec2 = vector_rep(text1, corpus_dict)
print(vec1)
print(vec2)
f.close()
```
