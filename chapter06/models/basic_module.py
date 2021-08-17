# 封装 nn.Module 部分功能（save&load）

import torch as t
import time

class BasicModule(t.nn.Module):

    def __init__(self):
        super(BasicModule,self).__init__() # t.nn.Module.__init__(self)
        self.model_name = str(type(self))

    def load(self,path):
        self.load_state_dict(t.load(path)) # load_state_dict()

    def save(self,name=None):
        if name is None:
            prefix = './checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')

        t.save(self.state_dict(),name)
        return name