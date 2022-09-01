import jieba
import torch


def edmundson(packed):
    """
    输入为打包的p和y，p，y要预先转换为字符串

    Example:
        无

    Args:
        :param packed:打包好的句子对（TUPLE）

    Return:
        :return: 均分/得分百分数
    """
    if isinstance(packed, tuple):
        total = 0
        p = packed[0]
        y = packed[1]
        for i ,j in zip(p, y):
            print("predicted:", i, "\n", "Tagedt:", j, "\n")
            answer = input("Score>>>0,1,2,3:")
            total += answer
    else:
        raise RuntimeError("INPUT error!Input should be a tuple")
    return total/(len(packed[0])), total/(5*len(packed[0]))

def rouge1(packed, language):
    """
    输入打包后的预测和标签、语言得到相对于的ROUGE1分数

    Example:
    （['常用分词工具‘, '分词粒度的控制],['接在自定义词典中中设置 词的频次', '代码具体如下'])

    Args:
        :param packed:预测和标签作为tuple输入
        :param language:English/Chinese
    Return:
        :return:rouge1分数
    """
    if not isinstance(packed, tuple):
        raise RuntimeError("INPUT error!Input shoule be a tuple")
    if language == "English":
        p = packed[0]
        y = packed[1]
        pTokens = []
        yTokens = []
        T = 0
        A = 0
        for i, j in zip(p, y):
            pTokens.append(i.split())
            yTokens.append(j.split())
        for pSen, ySen in zip(pTokens, yTokens):
            for pToken, yToken in zip(pSen, ySen):
                if pToken != 'O' and pToken == yToken:
                    T += 1
            A += len(pSen)
        return T/A
    else:
        p = packed[0]
        y = packed[1]
        pTokens = []
        yTokens = []
        T = 0
        A = 0
        for pSen, ySen in zip(p, y):
            pToken = list(jieba.cut(pSen, use_paddle=True))
            yToken = list(jieba.cut(ySen, use_paddle=True))
            A += len(pToken)    #利用预测长度来计算
            while len(pToken) > len(yToken):
                yToken += "O"
            while len(pToken) < len(yToken):    #保证句子长度不同或者标点出现的时候得到相同的padding长度来循环
                pToken += "O"
            pTokens.append(pToken)
            yTokens.append(yToken)
        for pSen, ySen in zip(pTokens, yTokens):
            for i, j in zip(pSen, ySen):
                if i != "O" and j != "O":
                    if i == j:
                        T += 1
        return T/A

def rouge2(packed, language):
    """
    输入打包的句子对，返回ROUGE2分数

    Example;
        无

    Args:
        :param packed: 打包的预测句子和真实句子（tuple）
        :param language: 语言(English/Chinese)
    Return:
        :return:
    """
    if language == "English":
        pass

def winsplitEn(inputs, win):
    """
    输入一个句子列表，返回一个完成window分词的列表,用来对英文进行win分词处理

    Args:
        :params inputs:句子列表
        :params win:int，窗口大小，不能大于句子长度
    Return:
        完成按照window分词的嵌套列表
    """
    if not isinstance(inputs, list):
        raise RuntimeError("Input shoule be a list!")
    tokens = []
    for i in inputs:
        tokens.append(sentenceSplit(i))
    splited = []
    splitedSen = []
    for sen in tokens:
        for i in range(len(sen) - win + 1):    # 得到的嵌套列表个数为length_of_the_sentence - window + 1
            splitedSen.append(sen[i:i+win])
        splited.append(splitedSen)
    return splited

def winsplitCh(inputs, win):
    """
    输入一个句子列表，返回一个完成window分词的列表，用来对中文做win分词梳理

    Args
        :param inputs: 句子列表
        :param win: 窗口大小，不能大于句子长度
    Return:
        返回按照window分词的嵌套列表
    """

def sentenceSplit(sentence):
    out = []
    out = sentence.split()
    return out

def acc_metrics(sentence_preds, y_labels):
    """
    output from the network
    sentence_preds:2 dimension tensor, size=[batch_size, padden_sentence_length]
    evaluation target
    y:2 dimension tensor, size=[batch_size, padden_sentence_length]
    Precision = (True_preds)/(True_preds+ False_preds)
    """
    Titems = 0
    Fitems = 0
    Totalitems = 0
    if sentence_preds.ndim != 2:
        raise RuntimeError("Size Error!The input shoule be [prediction, y] and both of them should be 3 dimensions")
    if y_labels.ndim != 2:
        raise RuntimeError("Size Error!The input shoule be [prediction, y] and both of them should be 3 dimensions")
    if sentence_preds.size()[1] != y_labels.size()[1]:
        raise RuntimeError("Both seqtence should have the same padding length")
    for sentence, y in zip(sentence_preds, y_labels):
        if sentence.size()[0] != y.size()[0]:
            raise RuntimeError("Both data should have the same batch_size")
        if torch.equal(sentence, y):
            Titems = Titems +1
        else:
            Fitems = Fitems +1
        Totalitems = Titems + Fitems
        """for i,j in zip(sentence, y):
            if i == j:
                Titems = Titems + 1
            else:
                Fitems = Titems + 1
        Totalitems = Titems + Fitems"""
    return Titems/Totalitems
