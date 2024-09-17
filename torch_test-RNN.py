# coding: utf-8

from __future__ import print_function

import os
import tensorflow.keras as kr
import torch
from torch import nn
from cnews_loader import read_category, read_vocab
from torch_model import TextCNN,TextRNN
from torch.autograd import Variable
import numpy as np
try:
    bool(type(unicode))
except NameError:
    unicode = str

base_dir = 'cnews'
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

class CnnModel:
    def __init__(self):
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.model = TextCNN()
        #self.model.load_state_dict(m_state_dict)
        self.model.load_state_dict(torch.load('20Rnn.pt'))

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]
        data = kr.preprocessing.sequence.pad_sequences([data], 600)
        data = torch.LongTensor(data)
        y_pred_cls = self.model(data)
        print(y_pred_cls)
        class_index = torch.argmax(y_pred_cls[0]).item()
        return self.categories[class_index]


class RnnModel:
    def __init__(self):
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.model = TextRNN()
        #self.model.load_state_dict(torch.load('20Rnn.pt'))
        self.model.load_state_dict(torch.load('model_params.pkl'))

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]
        data = kr.preprocessing.sequence.pad_sequences([data], 600)
        data = torch.LongTensor(data)
        y_pred_cls = self.model(data)
        class_index = torch.argmax(y_pred_cls[0]).item()
        return self.categories[class_index]



if __name__ == '__main__':
    #model = CnnModel()
    model = RnnModel()
    test_demo = ['热火vs骑士前瞻：皇帝回乡二番战 东部次席唾手可得新浪体育讯北京时间3月30日7:00',
'CBA季后赛半决赛，坐镇主场深圳队98-81击败浙江队，总比分2-1领先',
'配套设施建设方面，通知提到，加大保障性住房市政基础设施建设支持力度，各区政府对项目周边配套设施问题突出的区域应重点调度，建设周期内全程把控。应重点关注集体土地租赁住房项目周边市政配套建设问题，对于土地公开交易条件中明确由建设单位建设的，应加强监督管理，严格按照约定条件实施。',
'由于新房成交有一定滞后性，一般会从二手房成交来感知市场活跃度。上周北京二手住宅成交3761套，环比上涨18%',
 'IKEA 宜家家居是来自瑞典的全球知名的家具和家居用品零售商。',
'如今家居卖场各方市场力量正在涌动，卖场人流量呈现分散、流通渠道的多元化的现象',
'90所大学撤销这个专业？学科专业调整应避免“一窝蜂”跟进',
'考研生质疑招生人数和实际录取有出入 常州大学回应',
'三星ST550以全新的拍摄方式超越了以往任何一款数码相机',
'我国科学家首次在实验中实现模式匹配量子密钥分发',
'时隔半年，Valentino 2023春夏高级定制系列重返巴黎，将塞纳河畔的一处古典建筑改造为本次发布会的秀场',
'你无论潮流怎么试着玩新花样，总有一些人走在花样的反面。而且还有一种 “没有花样，却能让人一看再看” 的魔法。',
'截止收盘，沪指报3323.27点，涨1.14%，成交额为5149亿元；深成指报11338.67点，涨1.08%，成交额为6020亿元；创指报2324.72点，涨0.76%，成交额为2836亿元。',
'中邮人寿一季度净亏25亿元，最近一期风险综合评级结果下降',
'4月份，生产指数为50.2%，较上月下降4.4个百分点，显示制造业生产恢复势头稳中趋缓，但供给冲击压力持续缓解势头未变。与此相应，企业原材料采购量下降，从业人员数量减少。',
'据《海南日报》报道，4月29日晚，海南省委书记冯飞在海口以“四不两直”方式暗访调研“五一”假期社会治安、安全生产工作和旅游消费市场情况，看望慰问一线值班值守人员',
'在我们的众多免费在线动作小游戏中体验射击，戳刺，飞翔，奔向胜利！ 选择一个免费动作小游戏, 尽享快乐时光 你今天要玩什么呢？通过混沌晶石穿过通道，便可以进入神秘的混沌秘境。',
'充满史实感的回合制战略游戏 《全面坦克战略官》现已登陆Steam',
'这对于国内市场来说，插混增长率高于电动车。加推插混版本，广汽本田等于将这款经典车型同时放置到燃油赛道和新能源赛道',
'在国内紧凑型车市场，长安汽车多款等车型出色的产品实力获得了不俗的市场成绩。',]
    for i in test_demo:
        print(i,":",model.predict(i))


