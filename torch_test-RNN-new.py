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
       # self.model.load_state_dict(torch.load('20Rnn.pt'))
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
    test_demo = [
'三星ST550以全新的拍摄方式超越了以往任何一款数码相机',
'热火vs骑士前瞻：皇帝回乡二番战 东部次席唾手可得新浪体育讯北京时间3月30日7:00',
'CBA季后赛半决赛，坐镇主场深圳队98-81击败浙江队，总比分2-1领先',
'IKEA 宜家家居是来自瑞典的全球知名的家具和家居用品零售商。',
'如今家居卖场各方市场力量正在涌动，卖场人流量呈现分散、流通渠道的多元化的现象',
'来自 360测一测,你是否符合报考家庭教育指导师!在线咨询,了解更多详情',
'对这项工作，从国家到地方都高度重视。早在2017年，教育部、国家体育总局就联合发布了《关于推进学校体育场馆向社会开放的实施意见》，要求把学校体育场馆开放作为贯彻落实《“健康中国2030”规划纲要》和《全民健身条例》的重要举措，积极推进学校体育场馆向学生和社会开放',
'我国科学家首次在实验中实现模式匹配量子密钥分发',
'磁浮列车是一种靠磁浮力来推动的列车，由于其轨道的磁力使列车悬浮在空中，运行时不需接触地面，只受空气阻力影响，大大减少了车轮与轨道间的摩擦力，因此高速磁浮列车的速度可达每小时400公里以上',
'时隔半年，Valentino 2023春夏高级定制系列重返巴黎，将塞纳河畔的一处古典建筑改造为本次发布会的秀场',
'你无论潮流怎么试着玩新花样，总有一些人走在花样的反面。而且还有一种“没有花样，却能让人一看再看”的魔法。',
'居民可支配收入的细致结构以及随时间的变化，可以反映经济发展，可以反映贫富差距，也可以为制定合理的经济政策、民生政策提供依据。',
'纪念北京冬奥会成功举办一周年系列活动启动仪式',
'在我们的众多免费在线动作小游戏中体验射击，戳刺，飞翔，奔向胜利！ 选择一个免费动作小游戏, 尽享快乐时光 你今天要玩什么呢？通过混沌晶石穿过通道，便可以进入神秘的混沌秘境。',
'充满史实感的回合制战略游戏 《全面坦克战略官》现已登陆Steam',
'【剧集总数】52集 玛雅是一只与众不同的小蜜蜂，一出生就对身边的一切充满好奇。她不能理解蜜蜂世界多如牛毛的规则，也不想只是乖乖遵守规定。',
'2015年停产的迈凯伦旗舰P1采用的是一套插电混动系统，最大输出功率904马力，车重1488千克。',
'在国内紧凑型车市场，长安汽车凭借旗下锐程PLUS、逸动PLUS以及UNI-V等车型出色的产品实力获得了不俗的市场成绩。',
]
    for i in test_demo:
        print(i,":",model.predict(i))


