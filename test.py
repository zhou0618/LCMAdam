# import torch
# print(torch.__version__)
# print('GPU:',torch.cuda.is_available())#cuda是否可用
# print(torch.cuda.device_count())#返回GPU的数量
# torch.cuda.get_device_name(0)#返回gpu名字，设备索引默认从0开始
# print(torch.version.cuda)
# print(torch.backends.cudnn.version())
# x = torch.rand(5, 3)
# print(x)
import torch
print(torch.__version__)
print('GPU:',torch.cuda.is_available())#cuda是否可用
print(torch.cuda.device_count())#返回GPU的数量
torch.cuda.get_device_name(0)#返回gpu名字，设备索引默认从0开始
print(torch.version.cuda)
print(torch.backends.cudnn.version())
x = torch.rand(5, 3)
print(x)
# import torch
# from torch import nn
# logits = nn.MSELoss()
# logits = nn.CrossEntropyLoss()