import sys
from torch.utils.data import Dataset
sys.path.append("..")
from utils.data_process import pickleRead, padToMaxlength, selectList


class DatasetBase(Dataset):
    def __init__(self, tokens_list_en, tokens_list_zh):
        """初始化，为其添加各种必要属性

            Args:
                tokensListEN:经过fileTolist打开后的返回列表,语言为英文
                tokensListZH:经过fileTolist打开后返回的列表，语言为中文
        """
        super(DatasetBase, self).__init__()
        self.EN = tokens_list_en
        self.ZH = tokens_list_zh

    def __getitem__(self, item):
        """
        处理句子，返回一个输出和它的index

        item:
        Return:
            一个列表和一个y（列表）,两项数据都是字符串类型的
        """
        return self.EN[item], self.ZH[item]    # 准备的是英文->中文的训练数据

    def __len__(self):
        """
        返回数据集长度

        Return:
            数据集长度
        """
        return len(self.EN)


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
    datasetEN = pickleRead("./hot_data/data_dict/ENdataset.pkl")
    datasetZH = pickleRead("./hot_data/data_dict/ZHdataset.pkl")
    X, Y = selectList(data)
    batchedX = padToMaxlength(X, datasetEN, ify=False)
    batchedY = padToMaxlength(Y, datasetZH, ify=True)
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
    datasetEN = pickleRead("./hot_data/data_dict/ENdataset.pkl")
    datasetZH = pickleRead("./hot_data/data_dict/ZHdataset.pkl")
    X, Y = selectList(data)
    batchedX = padToMaxlength(X, datasetEN, 40)
    batchedY = padToMaxlength(Y, datasetZH, 40)
    """if len(batchedX[0]) > len(batchedY[0]):    # 暂用，需要继续学习packedsequence做mask操作之后再取消
        batchedY = padToMaxlength(batchedY, datasetZH)
    if len(batchedX[0]) < len(batchedY[0]):
        batchedX = padToMaxlength(batchedX, datasetEN)"""
    batchedData = (batchedX, batchedY)
    return batchedData

