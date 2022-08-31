import random
import torch


gen = random.random()
output = gen    # gen变量作为生成器，每次调用都产生一个随机数，作为一个概率值


class Generator:
    """临时生成器"""
    def __init__(self, batch_size, num_tag, seq_length):
        """规定有关生成大小的类属性"""
        self.batch_size = batch_size
        self.num_tag = num_tag
        self.seq_length = seq_length

    def generate(self):
        return torch.randn([self.batch_size, self.num_tag, self.seq_length], dtype=torch.float16)


class BeamSearch:
    """beam search类，调用类方法可以实现一次性的集束搜索"""
    def __init__(self, topk, num_beam, depth, gener):
        """
        类初始化
        Args:
            topk: 最多保留数
            num_beam: 集束数量
            depth: 深度
            gener: 生成器，在模型中表示生成字符的decoder，
                生成器的生成格式为：[batch_size, 1, num_tags]
                可以不经过softmax
        """
        self.topK = topk
        self.num_beam = num_beam
        self.depth = depth
        self.generator = gener

    def search(self):
        """
        进行集束搜索
        Return:
            完成集束搜索的序列
        """


class BeamContainer:
    """集束容器，每个集束用一个容器装载，用来进行打分或者扩增、抑制等操作"""
    def __init__(self, init_text_beam, init_pro_beam,  pad_id, EOS_id, SOS_id):
        """beam--->size:[batch_size, seq_length]"""
        self.text_beam = init_text_beam    # list
        self.pro_beam = init_pro_beam    # tensor
        self.pad_id = pad_id
        self.EOS_id = EOS_id
        self.SOS_id = SOS_id

    def add_element(self, element):
        """
        向集束添加下一组（个）概率值
        element---->size:[batch_size, 1]
        """
        self.text_beam = CatList(self.text_beam, element[0])
        self.pro_beam = torch.cat((self.pro_beam, element[1]), dim=1)

    def compute_scores(self):
        """beam分数计算，基础版分数计算"""
        scores = torch.zeros([self.pro_beam.size()[0], 1])
        for i in range(self.pro_beam.size()[0]):    # 0--->batch_size - 1
            for j in range(self.pro_beam.size()[1]):
                scores[i] *= self.pro_beam[i][j]



def CatList(list1, list2):
    """
    连接两个列表，列表为二维列表

    Return:
        连接好的列表，二维
    """
    out = []
    for i, j in zip(list1, list2):
        temp = i + j
        out.append(temp)
    return out