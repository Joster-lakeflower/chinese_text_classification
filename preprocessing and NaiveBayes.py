## 文本数据读取与预处理，并用朴素贝叶斯分类器训练与测试

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.scorer import make_scorer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pandas as pd
import numpy as np

#函数定义

def get_one_poetry(file,key = 'paragraphs'):
    #这个函数用于从一个json文件中读取诗词文本，返回一个由句子组成的列表
    sentences = []
    authors = []
    f = open(file,'r',encoding='utf-8')
    data = json.load(f)
    for works in data:
        content = works[key]
        author = works['author']
        for u in content:
            sentences.append(u)
        authors.append(author)
    f.close()
    return sentences,authors

def get_all_poetry(folder,prefix,end_num):
    #这个函数用于从文件目录中读取所有的json文件
    corpus = []
    authors = []
    
    for i in range(0, end_num+1000, 1000):
        path = folder + "/"+ prefix + str(i) + ".json"
        if(i == end_num):
            print(path)  #检查是否读完每一类文本
        sentences, this_authors = get_one_poetry(path)
        for x in sentences:
            corpus.append(x)
        for y in this_authors:
            authors.append(y)
    return corpus,authors

#读取词
def get_ci():
    #（可更改）ci_path为词的目录 
    ci_path = "chinese-poetry-master/ci" #相对路径
    ci, ci_authors = get_all_poetry(ci_path,"ci.song.",21000)
    supplement,supple_author = get_one_poetry("chinese-poetry-master/ci/ci.song.2019y.json") #存在一个不符合命名规律的文件
    for p in supplement:
        ci.append(p)
    for q in supple_author:
        ci_authors.append(q)
    return ci, ci_authors

#读取诗
def get_poem():
    #（可更改）poem_path为诗的目录 
    poem_path = "chinese-poetry-master/json" #相对路径

    poems = []
    authors = []
    song_poem, song_authors = get_all_poetry(poem_path,"poet.song.",254000) #宋诗
    tang_poem, tang_authors = get_all_poetry(poem_path,"poet.tang.",57000)  #唐诗
    for m in tang_poem:
        poems.append(m)
    for n in song_poem:
        poems.append(n)
    for o in song_authors:
        authors.append(o)
    for p in tang_authors:
        authors.append(p)
    return poems, authors

#这个函数根据data（由句子构成的列表）生成词云，将图片命名为saveloc
def get_word_cloud(data,saveloc):
    from wordcloud import WordCloud
    import jieba
    from collections import Counter

    words = []
    for sentence in data:
        temp = jieba.lcut(sentence)  #分词
        for word in temp:
            words.append(word)
    freq_dict = dict(Counter(words)) #生成词频字典

    wc = WordCloud()
    wc.generate_from_frequencies(freq_dict) #生成词云
    wc.to_file(saveloc) #保存图片

#这个函数根据authors（由作者构成的列表）生成词云，将图片命名为saveloc
def get_author_word_cloud(authors,saveloc):
    from wordcloud import WordCloud
    from collections import Counter

    wc = WordCloud()
    freq_dict = dict(Counter(authors)) #生成词频字典
    wc.generate_from_frequencies(freq_dict) #生成词云
    wc.to_file(saveloc) #保存图片

#这个函数用于一个列表的繁简转换
def to_simplified(list_of_sentences):
    import opencc
    cc = opencc.OpenCC('t2s') #繁体转简体
    result = []
    for sentence in list_of_sentences:
        result.append(cc.convert(sentence))
    print('繁简转换成功')
    return result

#这个函数用于将df(pandas)中的数据存入mysql中db_name数据库的to_table表
def save_data_to_mysql(df,to_table,username,passwd,db_name,host='127.0.0.1',port = '3306'):
    import pandas as pd
    from sqlalchemy import create_engine
    import sqlalchemy

    engine = create_engine('mysql+mysqlconnector://' + username + ':' + passwd + '@' + host + ':' + port + '/' + db_name)
    print("数据库"+db_name+"连接成功")
    df.to_sql(to_table,engine,if_exists='replace')
    print('文本数据保存到表'+to_table+'成功')
    return

#读取未经处理的原始数据，删除换行符和不必要的特殊符号
def read_raw_data(path,coding='utf-8'):
    data = []
    #这里是要处理的换行符和特殊符号
    delete = ['\u3000\u3000','\n','☆','★','○','◎','□',\
        '    ?','--------------------------------------------------------------------------------',\
          '==============================================================================']
    with open(path,encoding=coding) as f:
        lines = f.readlines()
        f.close()
    for line in lines:
        for x in delete:
            line = line.replace(x,"")
        if line == "":
            continue
        data.append(line)
    print('读取文件'+path+'成功')
    return data

#这个函数用于从MYSQL的db_name数据库中读取table表，返回一个df(pandas)
def read_data_from_mysql(table,username,passwd,db_name,host='127.0.0.1',port = '3306'):
    import pandas as pd
    import pymysql

    conn = pymysql.connect(	
        host = host,
        port = int(port),
        user = username,	
        passwd = passwd,	
        db = db_name,	
        charset = 'utf8')	

    df = pd.read_sql('select * from ' + table, conn)
    print('读取数据表'+table+'成功')
    print(len(df))
    return df

#这个函数用于中文分字
def ch_split(sentence):
    import re
    #提取出非汉字字符
    pattern_1 = re.compile(r'([\W])')
    parts = pattern_1.split(sentence)
    parts = [p for p in parts if len(p.strip())>0]
    # 分割中文
    pattern_2 = re.compile(r'([\u4e00-\u9fa5])')
    chars = pattern_2.split(sentence)
    chars = [w for w in chars if len(w.strip())>0] #由字符组成的列表
    
    return chars

'''
   这个函数用于将DataFrame中的数据进行细粒度的处理，分句，
   在每两个汉字之间添加空格，并在句子的开头和结尾加上<START>和<END>
'''
def split_sentences(df):
    import re
    delimeters = ['。','！','？','!','?','。”','！”','？”','?”','!”','! ','? ']
    sentences_segmented = []

    for i in range(len(df)):
        sentence = df.iloc[i,1]
        sentence = sentence.replace("（？）","")
        temp = re.split(r'(。”|！”|？”|\?”|!”|。|！|？|!|\?|!\u0020|\?\u0020)',sentence) #分句   
        for j in range(len(temp)):
            if len(temp[j]) > 0:
                if temp[j] in delimeters: #不考虑单个标点符号
                    continue
                tmp = ch_split(temp[j])   #由单个汉语字符组成的列表
                tmp2 = " ".join(tmp)      #添加了空格的句子
                if len(tmp2) > 0:
                    tmp2 = "<START> " + tmp2 + " <END>" #添加句子起始符和结束符
                    sentences_segmented.append(tmp2)
    return sentences_segmented

#这个函数用于读取数据库，返回每一类文本的内容与标签(DataFrame)
def get_train_data():
    import pandas as pd

    df_poem = read_data_from_mysql(table = 'poem', username = 'root', passwd = 'root', db_name = 'classification_corpus')
    df_ci = read_data_from_mysql(table = 'ci', username = 'root', passwd = 'root', db_name = 'classification_corpus')
    df_wyw = read_data_from_mysql(table = 'wyw', username = 'root', passwd = 'root', db_name = 'classification_corpus')
    df_news = read_data_from_mysql(table = 'news', username = 'root', passwd = 'root', db_name = 'classification_corpus')
    df_journal = read_data_from_mysql(table = 'journal', username = 'root', passwd = 'root', db_name = 'classification_corpus')
   
    poem = split_sentences(df_poem) #分句
    ci = split_sentences(df_ci)
    wyw = split_sentences(df_wyw)
    news = split_sentences(df_news)
    journal = split_sentences(df_journal)

    poem_label = ['诗'] * len(poem) #生成标签
    ci_label = ['词'] * len(ci)
    wyw_label = ['文言文'] * len(wyw)
    news_label = ['新闻'] * len(news)
    journal_label = ['期刊'] * len(journal)
 
    func1 = {'文本':poem, '类别':poem_label} #构建映射关系
    func2 = {'文本':ci, '类别':ci_label}
    func3 = {'文本':wyw, '类别':wyw_label}
    func4 = {'文本':news, '类别':news_label}
    func5 = {'文本':journal, '类别':journal_label}
    
    df1 = pd.DataFrame(func1) #生成含有文本和标签的DataFrame
    df2 = pd.DataFrame(func2)
    df3 = pd.DataFrame(func3)
    df4= pd.DataFrame(func4)
    df5 = pd.DataFrame(func5)
    print("文本和类别组合成功")

    return df1,df2,df3,df4,df5

#训练集特征选择
def extract_features(data):
    model = CountVectorizer(ngram_range=(2,2),token_pattern=r'((?u)\b\w+\b)')  # bag of words + bigram
    #model = TfidfVectorizer(ngram_range=(2,2),token_pattern=r'((?u)\b\w+\b)') # Tf-idf + bigram
    fi = model.fit_transform(data)  #进行特征选择后的词频矩阵

    return fi

#划分训练集、验证集、测试集
def split_data(df1,df2,df3,df4,df5):
    df = pd.concat([df1,df2,df3,df4,df5],axis=0) #拼接数据   
    print("五类数据拼接完成")
    print(df) 
    df = df.sample(frac=1,random_state=21) #打乱数据的次序
    
    train_data = extract_features(list(df['文本'])) #文本
    train_label = np.array(df['类别']) #标签

    train_data, test_data, train_label, test_label = train_test_split(train_data,
                                                                      train_label,
                                                                      test_size=0.02,
                                                                      random_state=21)
    #训练集、验证集和测试集的比例为98:1:1 
    validation_data, test_data, validation_label, test_label = train_test_split(test_data,
                                                                                test_label,
                                                                                test_size=0.5,
                                                                                random_state=21)
    return train_data, validation_data, test_data, train_label,\
           validation_label, test_label
  
#模型拟合
def categorize(train_data, train_label, test_data, test_label):
    clf = MultinomialNB().fit(train_data, train_label)     #朴素贝叶斯分类器
    predicted = clf.predict(test_data)
    print("测试数据前十条：",test_data[:10])
    print("测试标签前十条：",test_label[:10])
    print("预测标签前十条：",predicted[:10])
    print(metrics.classification_report(test_label, predicted)) #指标评估


#主程序

#从文件目录中获得诗词文本和作者
corpus_poem, poem_authors = get_poem()
corpus_ci, ci_authors = get_ci()

#将诗、词存入DataFrame和Mysql
df_ci = pd.DataFrame(to_simplified(corpus_ci)) 
df_poem = pd.DataFrame(to_simplified(corpus_poem))
#save_data_to_mysql(df = df_ci, to_table = 'ci', username = 'root', passwd = 'root', db_name = 'classification_corpus')
#save_data_to_mysql(df = df_poem, to_table = 'poem', username = 'root', passwd = 'root', db_name = 'classification_corpus')

#生成词云（作者和高频字词）
get_author_word_cloud(authors=ci_authors,saveloc='ci_authors.png')
get_author_word_cloud(authors=poem_authors,saveloc='poem_authors.png')
get_word_cloud(data=corpus_ci,saveloc='ci.png')
get_word_cloud(data=corpus_poem,saveloc='poem.png')

#读取文言文、新闻和期刊论文
wyw = read_raw_data('文言文数据.txt')               #文言文
news = read_raw_data('data.txt',coding='ansi')     #新闻
journal = read_raw_data('期刊论文文本.txt')         #期刊论文

#将文言文、新闻和期刊论文存入数据库，这里的文本只经过了粗粒度的处理，有的是一句话，有的是一段话
df_wyw = pd.DataFrame(to_simplified(wyw)) 
df_news = pd.DataFrame(to_simplified(news))
df_journal = pd.DataFrame(to_simplified(journal))
#save_data_to_mysql(df = df_wyw, to_table = 'wyw', username = 'root', passwd = 'root', db_name = 'classification_corpus')
#save_data_to_mysql(df = df_news, to_table = 'news', username = 'root', passwd = 'root', db_name = 'classification_corpus')
#save_data_to_mysql(df = df_journal, to_table = 'journal', username = 'root', passwd = 'root', db_name = 'classification_corpus')

a, b, c, d, e = get_train_data() #经过细粒度处理后的语料库，以句子为单位

#训练集、验证集、测试集
train_data, validation_data, test_data, train_label,\
    validation_label, test_label = split_data(a,b,c,d,e)

#使用朴素贝叶斯分类器进行分类
categorize(train_data, train_label,  validation_data, validation_label)