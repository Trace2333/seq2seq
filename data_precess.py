import os.path
import pickle
import logging
import numpy as np
from tqdm import tqdm
from gensim.models import word2vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
def fileTolist(filename):
    """
    未处理文件转为list

    Args:
        :param filename: 未处理数据
    Return:
        分词后句子，没有用jieba
    """
    enF = open(filename, "r", encoding="utf8")
    lines = enF.readlines()
    tokens = []
    for line in lines:
        t = line.split()
        tokens.append(t)
    enF.close()
    return tokens
def dictCreate(tokens, targetFile, tokens1=None, tokens2=None):
    """
    建立token->id词典文件

    Args:
        :param tokens: token构成的list
        :param targetFile: 输出的dataset pickle文件
    Return:
        无
    """
    if not os.path.exists(".\\datasets"):
        os.mkdir(".\\datasets")
    tokensList = []
    datasetDict = {}
    for sentence in tokens:
        tokensList.extend(sentence)
    if tokens1 is not None:
        for sentence in tokens1:
            tokensList.extend(sentence)
    if tokens2 is not None:
        for sentence in tokens2:
            tokensList.extend(sentence)
    tokensList = list(set(tokensList))
    tokensList.append("<EOS>")
    tokensList.insert(0, "<SOS>")
    tokensList.insert(0, "O")
    for token, i in zip(tokensList, range(len(tokensList))):
        if token not in datasetDict:
            datasetDict[token] = i
    if not os.path.exists(".\\datasets\\" + targetFile):
        with open(".\\datasets\\" + targetFile, 'wb') as f1:
            pickle.dump(datasetDict, f1)
        print(targetFile + "Created!!!")
    else:
        print(targetFile + "Exists!!!")

def sensToPaddedList(sens, dataset, forcedLen=None):
    """
    padding输入

    Args:
        :param sens: 输入的句子列表，由dataloader得到
        :param dataset: dataset文件
        :param forcedLen: 规定长度，default=列表的最大长度
    Return:
        sens个padding后的列表
    """
    padded = []
    paddedIds = []
    ids = []
    maxLength = max(len(list(x)) for x in sens)
    if forcedLen is None:
        for sen in sens:
            paddedSen = sen
            while len(sen) < maxLength:
                paddedSen.extend("O")
            padded.append(paddedSen)
    else:
        maxLength = forcedLen
        for sen in sens:
            paddedSen = sen
            while len(sen) < maxLength:
                paddedSen.extend("O")
            while len(sen) >maxLength:
                paddedSen.pop()
            padded.append(paddedSen)
    for i in padded:
        for j in i:
            id = dataset[j]
            ids.append(id)
        paddedIds.append(ids)
    return paddedIds

def pickleRead(filename):
    """
    读取dataset
    Args:
        :param filename: dataset文件名
    Return:
        无
    """
    f = open(filename, "rb")
    dataset = pickle.load(f)
    f.close()
    return dataset

def word2vecModelCreate(fileName, targetFilename):
    """
    利用词表训练word2vec模型
    Args:
        :param fileName: 未处理文件路径
        :param targetFilename: 目标的Word2vec向量文件名
    Return:
        无
    """
    tokensList = fileTolist(fileName)
    if not os.path.exists(".\\word2vec"):
        os.mkdir(".\\word2vec")
    if os.path.exists(".\\word2vec"):
        model = word2vec.Word2Vec(
            sentences=tokensList,
            vector_size=300,
            window=3,
            min_count=0,
            epochs=100
        )
        model.wv.save_word2vec_format(targetFilename)

def loadBinVector(binvec, dataset):
    """
    查阅预训练词表来得到词表中的词嵌入，词表中没有的词无法得到词嵌入,保存词嵌入矩阵文件

    Args:
        binvec: 文件名或路径，指示预训练词嵌入文件
        dataset: 词表，token->id
     Return:
         无
    """
    matrix = {}
    n = 0
    m = 0
    with open(binvec, "rb") as f:
        iteration = tqdm(f, desc="Precessing on BinVecs....")
        for line in iteration:
            m += 1
            if m == 1:
                continue
            word = line.decode().split()[0]
            vec = np.asarray(line.decode().split()[1:], dtype=np.float)
            if word in dataset:
                matrix[dataset[word]] = vec
            n += 1
            #if n % 10000 == 0:
                #print(f"Precessed {n} Words")
        return matrix

def addUnknownWords(dataset, matrix, targetFile):
    """
    添加在预训练词向量中没有的词到词嵌入表中，该词向量是随机生成的

    Args:
        dataset: dataset文件
        matrix: 词嵌入矩阵（经过loadBinVec函数处理的）
        targetFile: 想要保存的文件名
    Return:
        无
    """
    if not os.path.exists(".\\embw"):
        os.mkdir(".\\embw")
    for word in dataset:
        if dataset[word] not in matrix:
            matrix[dataset[word]] = np.asarray(np.random.uniform(-0.25, 0.25, 300), dtype=np.float32)
    if not os.path.exists(".\\embw\\" + targetFile):
        with open(".\\embw\\" + targetFile, 'wb') as f1:
            pickle.dump(matrix, f1)
        print(targetFile + "Created!!!")
    else:
        print(targetFile + "Exists!!!")

def dictTondarray(embw):
    embedding = np.zeros((len(embw) + 1, 300), dtype=np.float)
    for i in embw:
        embedding[i] = embw[i]
    return embedding

def padToMaxlength(inputs, dataset, forced_length=None):
    """
    将句子列表padding
    Args:
        inputs: 输入的句子列表，已经分词,由tuple组成
        dataset: 词典
        forced_length: 指定的最大长度，没有给默认按照列表中的最大句子长度分词
    Return:
        分词后的嵌套列表
    """
    inputList = []
    sen = []
    datasetLen = len(dataset)
    max_length = max(len(list(x)) for x in inputs) + 2
    for i in inputs:
        for j in i:
            sen.append(dataset[j])
        sen.append(datasetLen - 1)
        sen.insert(0, 1)
        inputList.append(sen)
        sen = []
    if forced_length is None:
        num_padded_length = max_length  # padding to the curant max length
        padded_list = []
        for sentence in inputList:
            padded_sentence = sentence
            while len(padded_sentence) < num_padded_length:
                padded_sentence.append(0)  # is the max length indefinitely
            padded_list.append(padded_sentence)
        return padded_list
    else:
        if max_length < forced_length:
            num_padded_length = forced_length
            padded_list = []
            for sentence in inputList:
                padded_sentence = sentence
                while len(padded_sentence) < num_padded_length:
                    padded_sentence.append(0)  # is the max length indefinitely
                padded_list.append(padded_sentence)
            return padded_list
        else:
            num_padded_length = forced_length  # some sentences shoule be cut.
            padded_list = []
            for sentence in inputList:
                padded_sentence = sentence
                if len(padded_sentence) == num_padded_length:
                    padded_list.append(padded_sentence)
                if len(padded_sentence) < num_padded_length:
                    while len(padded_sentence) < num_padded_length:
                        padded_sentence.append(0)  # is the max length indefinitely
                    padded_list.append(padded_sentence)
                if len(padded_sentence) > num_padded_length:
                    while len(padded_sentence) > num_padded_length:
                        padded_sentence.pop()
                    padded_list.append(padded_sentence)
            return padded_list

def selectList(data):
    X = []
    Y = []
    for i in data:
        X.append(i[0])
        Y.append(i[1])
    return X, Y

def collate_fn(data):
    """
    自定义collate function
    Args:
        data: 由dataloader返回的数据，自动填入，
    Note:
        返回的是tokens，不是id
    Return:
        经过处理后的批数据
    """
    datasetEN = pickleRead(".\\datasets\\ENdataset.pkl")
    datasetZH = pickleRead(".\\datasets\\ZHdataset.pkl")
    X, Y = selectList(data)
    batchedX = padToMaxlength(X, datasetEN)
    batchedY = padToMaxlength(Y, datasetZH)
    """if len(batchedX[0]) > len(batchedY[0]):    # 暂用，需要继续学习packedsequence做mask操作之后再取消
        batchedY = padToMaxlength(batchedY, datasetZH)
    if len(batchedX[0]) < len(batchedY[0]):
        batchedX = padToMaxlength(batchedX, datasetEN)"""
    batchedData = (batchedX, batchedY)
    return batchedData

def evalcollate_fn(data):
    """
    自定义collate function
    Args:
        data: 由dataloader返回的数据，自动填入，
    Note:
        返回的是tokens，不是id
    Return:
        经过处理后的批数据
    """
    datasetEN = pickleRead(".\\datasets\\ENdataset.pkl")
    datasetZH = pickleRead(".\\datasets\\ZHdataset.pkl")
    X, Y = selectList(data)
    batchedX = padToMaxlength(X, datasetEN, 40)
    batchedY = padToMaxlength(Y, datasetZH, 40)
    """if len(batchedX[0]) > len(batchedY[0]):    # 暂用，需要继续学习packedsequence做mask操作之后再取消
        batchedY = padToMaxlength(batchedY, datasetZH)
    if len(batchedX[0]) < len(batchedY[0]):
        batchedX = padToMaxlength(batchedX, datasetEN)"""
    batchedData = (batchedX, batchedY)
    return batchedData

def process():
    filenames = [
        ".\\cs_en\\train\\news-commentary-v13.zh-en.en",
        ".\\cs_en\\train\\news-commentary-v13.zh-en.zh",
        ".\\cs_en\\test\\newstest2017.tc.en",
        ".\\cs_en\\test\\newstest2017.tc.zh",
        ".\\cs_en\\dev\\newsdev2017.tc.zh",
        ".\\cs_en\\dev\\newsdev2017.tc.en"
    ]
    targetnames1 = [
        "trainEN.pkl",
        "trainZH.pkl",
        "testEN.pkl",
        "testZH.pkl",
        "devZH.pkl",
        "devEN.pkl"
    ]
    targetnames2 = [
        "trainEN.pkl",
        "trainZH.pkl",
        "testEN.pkl",
        "testZH.pkl",
        "devZH.pkl",
        "devEN.pkl"
    ]
    for i, j in zip(filenames, targetnames1):
        dictCreate(fileTolist(i), j)
    dictCreate(fileTolist(filenames[0]), "ENdataset.pkl", fileTolist(filenames[2]), fileTolist(filenames[5]))
    dictCreate(fileTolist(filenames[1]), "ZHdataset.pkl", fileTolist(filenames[3]), fileTolist(filenames[4]))

    trainZH = pickleRead(".\\datasets\\trainZH.pkl")
    trainEN = pickleRead(".\\datasets\\trainEN.pkl")
    testZH = pickleRead(".\\datasets\\testZH.pkl")
    testEN = pickleRead(".\\datasets\\testEN.pkl")
    EN = pickleRead(".\\datasets\\ENdataset.pkl")
    ZH = pickleRead(".\\datasets\\ZHdataset.pkl")
    matrix = loadBinVector("C:\\Users\\Trace\\Desktop\\Projects\\Keywords_Reaserch\\seq2seq\\embw\\sgns.wiki.word",
                           trainZH)
    addUnknownWords(trainZH, matrix, "TrainZHembw.pkl")
    matrix = loadBinVector("C:\\Users\\Trace\\Desktop\\Projects\\Keywords_Reaserch\\SRNN\\data\\original_data\\GoogleNews-vectors-negative300.txt",
                           trainEN)
    addUnknownWords(trainEN, matrix, "TrainENembw.pkl")
    matrix = loadBinVector("C:\\Users\\Trace\\Desktop\\Projects\\Keywords_Reaserch\\seq2seq\\embw\\sgns.wiki.word",
                           testZH)
    addUnknownWords(testZH, matrix, "EvalZHembw.pkl")
    matrix = loadBinVector("C:\\Users\\Trace\\Desktop\\Projects\\Keywords_Reaserch\\SRNN\\data\\original_data\\GoogleNews-vectors-negative300.txt",
                           testEN)
    addUnknownWords(testEN, matrix, "EvalENembw.pkl")
    matrix = loadBinVector("C:\\Users\\Trace\\Desktop\\Projects\\Keywords_Reaserch\\seq2seq\\embw\\sgns.wiki.word",
                           ZH)
    addUnknownWords(ZH, matrix, "ZHembw.pkl")
    matrix = loadBinVector(
        "C:\\Users\\Trace\\Desktop\\Projects\\Keywords_Reaserch\\SRNN\\data\\original_data\\GoogleNews-vectors-negative300.txt",
        EN)
    addUnknownWords(EN, matrix, "ENembw.pkl")