import torch.nn as nn

class ResNetBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.resnet = model
        del self.resnet.fc

    def forward(self, x):
        
        out = self.resnet.conv1(x)
        out = self.resnet.bn1(out)
        out = self.resnet.relu(out)
        out = self.resnet.maxpool(out)

        out = self.resnet.layer1(out)
        out = self.resnet.layer2(out)
        out = self.resnet.layer3(out)
        out = self.resnet.layer4(out)
        
        return out