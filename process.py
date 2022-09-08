from utils.data_process import dictCreate, fileTolist, pickleRead, loadBinVector, addUnknownWords


def process():
    """产出的处理文件的文件名只需要写名字即可，路径会自己创建"""
    filenames = [
        "./origin_data/cs_en/train/news-commentary-v13.zh-en.en",
        "./origin_data/cs_en/train/news-commentary-v13.zh-en.zh",
        "./origin_data/cs_en/test/newstest2017.tc.en",
        "./origin_data/cs_en/test/newstest2017.tc.zh",
        "./origin_data/cs_en/dev/newsdev2017.tc.zh",
        "./origin_data/cs_en/dev/newsdev2017.tc.en"
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

    trainZH = pickleRead("./hot_data/data_dict/trainZH.pkl")
    trainEN = pickleRead("./hot_data/data_dict/trainEN.pkl")
    testZH = pickleRead("./hot_data/data_dict/testZH.pkl")
    testEN = pickleRead("./hot_data/data_dict/testEN.pkl")
    EN = pickleRead("./hot_data/data_dict/ENdataset.pkl")
    ZH = pickleRead("./hot_data/data_dict/ZHdataset.pkl")
    matrix = loadBinVector("./origin_data/sgns.wiki.word",
                           trainZH)
    addUnknownWords(trainZH, matrix, "TrainZHembw.pkl")
    matrix = loadBinVector("./origin_data/GoogleNews-vectors-negative300.txt",
                           trainEN)
    addUnknownWords(trainEN, matrix, "TrainENembw.pkl")
    matrix = loadBinVector("./origin_data/sgns.wiki.word",
                           testZH)
    addUnknownWords(testZH, matrix, "EvalZHembw.pkl")
    matrix = loadBinVector("./origin_data/GoogleNews-vectors-negative300.txt",
                           testEN)
    addUnknownWords(testEN, matrix, "EvalENembw.pkl")
    matrix = loadBinVector("./origin_data/sgns.wiki.word",
                           ZH)
    addUnknownWords(ZH, matrix, "ZHembw.pkl")
    matrix = loadBinVector(
        "./origin_data/GoogleNews-vectors-negative300.txt",
        EN)
    addUnknownWords(EN, matrix, "ENembw.pkl")


if __name__ == '__main__':
    process()