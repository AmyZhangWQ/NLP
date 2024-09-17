import torch
t1 = torch.LongTensor(3, 5)
print(t1.type())
# 转换为其他类型
t2=t1.type(torch.FloatTensor)
print(t2.type())


torch.LongTensor
torch.FloatTensor
