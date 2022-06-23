import torch
import torch.nn as nn

from torch.optim.lr_scheduler import OneCycleLR
from positional_encodings.torch_encodings import PositionalEncodingPermute2D, Summer

from q2l_labeller.models.timm_backbone import TimmBackbone

class Query2Label(nn.Module):
    def __init__(
        self, model, conv_out, num_classes, hidden_dim=256, nheads=8, 
        encoder_layers=6, decoder_layers=6, use_pos_encoding=False):
        
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.use_pos_encoding = use_pos_encoding

        self.backbone = TimmBackbone(model) # outputs HW x 
        self.conv = nn.Conv2d(conv_out, hidden_dim, 1)
        self.transformer = nn.Transformer(
            hidden_dim, nheads, encoder_layers, decoder_layers)

        # prediction head
        self.classifier = nn.Linear(num_classes * hidden_dim, num_classes)

        # label parameters
        # TODO: Rename 
        self.query_pos = nn.Parameter(torch.rand(1, num_classes, hidden_dim))

    def forward(self, x):
        # produces output of shape [N x C x H x W]
        out = self.backbone(x)
        
        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(out)
        B, C, H, W = h.shape

        # add position encodings
        if self.use_pos_encoding:
            # returns the encoding object
            pos_encoder = PositionalEncodingPermute2D(C)

            # returns the summing object
            encoding_adder = Summer(pos_encoder)

            # input with encoding added
            h = encoding_adder(x)

        # convert h from [N x C x H x W] to [H*W x N x C] (N=batch size)
        # this corresponds to the [SIZE x BATCH_SIZE x EMBED_DIM] dimensions 
        # that the transformer expects
        h = h.flatten(2).permute(2, 0, 1)
        
        # image feature vector "h" is sent in after transformation above; we 
        # also convert query_pos from [1 x TARGET x (hidden)EMBED_SIZE] to 
        # [TARGET x BATCH_SIZE x (hidden)EMBED_SIZE]
        query_pos = self.query_pos.repeat(B, 1, 1)
        query_pos = query_pos.transpose(0, 1)
        h = self.transformer(h, query_pos).transpose(0, 1)
        
        # output from transformer is of dim [TARGET x BATCH_SIZE x EMBED_SIZE];
        # however, we transpose it to [BATCH_SIZE x TARGET x EMBED_SIZE] above
        # and then take an average along the TARGET dimension.
        #
        # next, we project transformer outputs to class labels
        #h = h.mean(1)
        h = torch.reshape(h,(B, self.num_classes * self.hidden_dim))

        return self.classifier(h)
        