import torch 
import torch.nn as nn
from resnet import ResNet18,ResNet101
import torch.functional as F


class L2Norm(nn.Module):
    def __init__(self,n_channels,scale):
        self.input = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weights = nn.parameters(torch.tensor(n_channels))
        self.reset_parameters()

        def reset_parameteres():
            nn.init.constant_(self.weights,self.gamma)
        
        def forward(self,x):
            norm = torch.sqrt(x.pow(2).sum(dim=1,keepdim = True)) + self.eps #https://mathworld.wolfram.com/L2-Norm.html 
            x = torch.div(x,norm)
            self.weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
            x= self.weight *x
            return x
        

def Extra(self):
    
    conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1)
    conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
    conv9_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
    conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
    conv10_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
    conv10_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
    conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
    conv11_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1)

    self.layers = [conv8_1, conv8_2, conv9_1, conv9_2, conv10_1, conv10_2, conv11_1, conv11_2]

    return self.layers


def Feature_extractor(ver, extral, bboxes, num_classes):
    
    loc_layers = []
    conf_layers = []
    
    if ver == 'RES18_SSD':
        loc_layers += [nn.Conv2d(128, bboxes[0] * 4, kernel_size=3, padding=1)]
        loc_layers += [nn.Conv2d(256, bboxes[1] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(128, bboxes[0] * num_classes, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, bboxes[1] * num_classes, kernel_size=3, padding=1)]
    
    elif ver == 'RES101_SSD' | ver=="RES50_SSD":
        loc_layers += [nn.Conv2d(512, bboxes[0] * 4, kernel_size=3, padding=1)]
        loc_layers += [nn.Conv2d(1024, bboxes[1] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(512, bboxes[0] * num_classes, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(1024, bboxes[1] * num_classes, kernel_size=3, padding=1)]
    
    
    for k, v in enumerate(extral[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, bboxes[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, bboxes[k]
                                  * num_classes, kernel_size=3, padding=1)]
        
    
    
    return loc_layers, conf_layers 

class RES18_SSD(nn.Module):
    def __init__(self,num_classes,bboxes,pretrain = None):
        super(RES18_SSD,self).__init__()
        self.ver = "RES18_SSD"
        self.num_classes = num_classes
        self.bboxes=bboxes
        self.extra_list = Extra()
        self.loc_layers_list , self.conf_layers_list = Feature_extractor(self.ver,self.extra_list,self.bboxes,self.num_classes)
        self.L2Norm = L2Norm(128,20)

        resnet = ResNet18()
        if pretrain:
            net = torch.load("your weights path")
            print("Loading Pretrained Model")
            resnet.load_state_dict(net)
        
        self.res = nn.Sequential(
            *list(resnet.children())[:-2],
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            nn.Conv2d(512,1024,kernel_size=3,padding=6,dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024,1024,kernel_size=1),
            nn.ReLU(inplace = True)
        )
        self.extras = nn.ModuleList(self.extra_list)
        self.loc = nn.ModuleList(self.loc_layers_list)
        self.conf = nn.ModuleList(self.conf_layers_list)

        def forward(self, x):
            source = []
            loc = []
            conf = []
            res_source = [5, 6]
            for i, v in enumerate(self.res):
                x = v(x)
                if i in res_source:
                    if i == 5:
                        s = self.L2Norm(x)
                    else:
                        s = x
                    source.append(s)
            for i, v in enumerate(self.extras):
                x = F.relu(v(x), inplace=True)
                if i % 2 == 1:
                    source.append(x)
            for s, l, c in zip(source, self.loc, self.conf):
                loc.append(l(s).permute(0, 2, 3, 1).contiguous())
                conf.append(c(s).permute(0, 2, 3, 1).contiguous())
            loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
            conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
            
            loc = loc.view(loc.size(0), -1, 4)
            conf = conf.view(conf.size(0), -1, self.num_classes)
            return loc, conf

class RES101_SSD(nn.Module):

    def __init__(self, num_classes, bboxes, pretrain=None ):
        super(RES101_SSD, self).__init__()

        self.ver = 'RES101_SSD'
        self.num_classes = num_classes
        self.bboxes = bboxes      
        self.extra_list = Extra()
        self.loc_layers_list, self.conf_layers_list = Feature_extractor(self.ver, self.extra_list, self.bboxes, self.num_classes)
        self.L2Norm = L2Norm(512, 20)


        resnet = ResNet101()
        if pretrain:
            net = torch.load('./weights/resnet101-5d3b4d8f.pth')
            print('resnet101 pretrain_model loading...')
            resnet.load_state_dict(net)
        
        self.res = nn.Sequential(
            *list(resnet.children())[:-2],
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(2048, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.extras = nn.ModuleList(self.extra_list)
        self.loc = nn.ModuleList(self.loc_layers_list)
        self.conf = nn.ModuleList(self.conf_layers_list)
        
    def forward(self, x):

        source = []
        loc = []
        conf = []
        res_source = [5, 6]
        for i, v in enumerate(self.res):
            x = v(x)
            if i in res_source:
                if i == 5:
                    s = self.L2Norm(x)
                else:
                    s = x
                source.append(s)

        for i, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if i % 2 == 1:
                source.append(x)


        for s, l, c in zip(source, self.loc, self.conf):
            loc.append(l(s).permute(0, 2, 3, 1).contiguous())
            conf.append(c(s).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)


       

        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)
        return loc, conf
    
    