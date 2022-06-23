import timm
import torch.nn as nn

class TimmBackbone(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        # Creating the model in this way produces unpooled, unclassified features
        self.model = timm.create_model(
            model_name, 
            pretrained=True,
            num_classes=0,
            global_pool='')        

    def forward(self, x):
        
        out = self.model(x)
        
        return out