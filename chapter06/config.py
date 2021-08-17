import warnings
import torch as t

class DefaultConfig(object):

    # visdom
    env = 'default'
    vis_port = 8097
    model = 'SqueezeNet' # 名字必须与models/__init__.py中的名字一致

    # 数据和模型路径
    train_data_root = 'E:/pytorch_workspace/Hands-on-learning-and-deep-learning/05_Practice/data/train/'
    test_data_root = 'E:/pytorch_workspace/Hands-on-learning-and-deep-learning/05_Practice/data/test1/'
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载

    #训练参数
    batch_size = 32
    use_gpu = True
    num_workers = 0
    print_freq = 20 # print info every N batch

    # log和result路径
    debug_file = './tmp/debug'
    result_file = 'result.csv'

    max_epoch = 10
    lr = 0.001
    lr_decay = 0.5 # when val_loss increase,lr = lr*lr_decay
    weight_decay = 0e-5

    def _parse(self, **kwargs):

        for k,v in kwargs.items():
            if not hasattr(self,k):
                warnings.warn('Warning: opt has no attribut %s' %k)
            setattr(self,k,v)

        opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')

        print('user config:')
        for k,v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k,getattr(self,k)) # getattr()返回对象的属性值



opt = DefaultConfig()