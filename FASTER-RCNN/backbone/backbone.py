import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleNeck(nn.Module):
    expansion =4

    def __init__(self,in_planes,planes,stride=1):
        super(BottleNeck,self).__init__() #super() for calling parent class constructor
        self.conv1 = nn.Conv2d(in_planes,planes,kernel_size=1,bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=False) #stride,padding are applied to reduce the size of image
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,planes*self.expansion,kernel_size=1,bias=False)
        self.bn3   = nn.BatchNorm2d(planes*self.expansion)

        self.downsample = nn.Sequential() #downsample is used to reduce the size of image

        if(stride!=1 or in_planes!=planes*self.expansion):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes,planes*self.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(planes*self.expansion)
            )   
        
    def forward(self,x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            out += self.downsample(x)
            out = F.relu(out)
            return out
        

class ResNet(nn.Module):
    def __init__(self,block,num_blocks,num_classes=2): #We are testing this on fire and smoke so 2 you may edit this for personal use cases
        super(ResNet,self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False) #3 is the number of channels in the image
        self.bn1   = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self._make_layer(block,64,num_blocks[0],stride=1)
        self.layer2 = self._make_layer(block,128,num_blocks[1],stride=2)
        self.layer3 = self._make_layer(block,256,num_blocks[2],stride=2)
        self.layer4 = self._make_layer(block,512,num_blocks[3],stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion,num_classes)

    def _make_layer(self,block,planes,num_blocks,stride): #This function is used to create the layers
        strides = [stride] + [1]*(num_blocks-1) #stride is applied to the first layer only
        layers = [] #This is the list of layers
        for stride in strides:#This loop is used to create the layers
            layers.append(block(self.in_planes,planes,stride))#This is the layer
            self.in_planes = planes*block.expansion#This is the number of input planes for the next layer
        return nn.Sequential(*layers) #This returns the layers
    
    def forward(self,x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out
    
def ResNet50():
    
    return ResNet(BottleNeck,[3,4,6,3])

def ResNet101():
    
    return ResNet(BottleNeck,[3,4,23,3])
    
if __name__ =="__main__":
    net = ResNet50()
    y = net(torch.randn(1,3,224,224))
    print(y.size())
    

