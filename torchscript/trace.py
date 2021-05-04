#coding=utf-8
import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
       super(MyModule,self).__init__()
       self.conv1 = nn.Conv2d(1,3,3)
    def forward(self,x):
       x = self.conv1(x)
       return x

model = MyModule()  # 实例化模型
trace_module = torch.jit.trace(model,torch.rand(1,1,224,224)) 
print(trace_module.code)  # 查看模型结构
output = trace_module (torch.ones(1, 1, 224, 224)) # 测试
print(output)
# trace_modult('model.pt') 
