#coding=utf-8
import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule,self).__init__()
        self.conv1 = nn.Conv2d(1,3,3)
        self.conv2 = nn.Conv2d(2,3,3)

    def forward(self,x):
        b,c,h,w = x.shape
        if c ==1:
            x = self.conv1(x)
        else:
            x = self.conv2(x)
        return x

model = MyModule()

# 这样写会报错，因为有控制流
# trace_module = torch.jit.trace(model,torch.rand(1,1,224,224)) 

# 此时应该用script方法
script_module = torch.jit.script(model) 
print(script_module.code)
output = script_module(torch.rand(1,1,224,224))
