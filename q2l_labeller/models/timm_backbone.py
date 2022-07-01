import timm
import torch.nn as nn


class TimmBackbone(nn.Module):
    """Specified timm model without pooling or classification head"""

    def __init__(self, model_name):
        """Downloads and instantiates pretrained model

        Args:
            model_name (str): Name of model to instantiate.
        """
        super().__init__()

        # Creating the model in this way produces unpooled, unclassified features
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=0, global_pool=""
        )

    def forward(self, x):
        """Passes batch through backbone

        Args:
            x (Tensor): Batch tensor

        Returns:
            Tensor: Unpooled, unclassified features from image model.
        """

        out = self.model(x)

        return out
