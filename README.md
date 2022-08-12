# Query2Label
My implementation of the [Query2Label](https://arxiv.org/abs/2107.10834) paper.

## Overview
This repository contains my from-scratch implementation of the Query2Label model, which was was based on FAIR's DETR model. Like the original paper implementation, mine achieves SOTA performance on multi-label image classification tasks. This repository is intended to serve both as a resource in learning this model as well as a performant, usable starting point for your own multi-label image tasks.

### Architecture
You can read about the architecture in detail in my [blog post](https://medium.com/@curttigges/building-a-transformer-powered-sota-image-labeller-cfe25e6d69f1) on the topic. In brief, the model consists of the following:

1. Images are passed through a pre-trained backbone (usually a ResNet variant or a Vision Transformer).
2. The backbone output is projected through a linear layer to reduce the number of feature planes (2048 from the larger ResNets) by some specified amount.
3. Position encodings are added to the feature planes and the result is reshaped to fit into the transformer.
4. The reshaped output is then passed to a simple, default transformer (the paper used one encoder layer and two decoder layers), along with learnable label embeddings. Backbone output is passed where one would normally pass source word embeddings for a language model transformer, and the embedded labels are passed where one would normally pass target word sequences. Masking is omitted since we are not outputting a sequence.
5. The output is passed through a classification head and softmaxed, which produces a tensor containing probabilities for each of the potential labels.

### Data
I used the MS COCO 2014 dataset for training, but this architecture works well with many similar datasets.

## Environment & Setup
This model was trained with the following packages:
- `pytorch 1.8.2`
- `torchvision 0.9.2`
- `pytorch-lightning 1.6.1`
- `torchmetrics 0.8.0`

## Repo Structure
### q2l_labeller/data
- coco_data_module.py - Data module for MS COCO 2014. This can be used to prepare annotations and transform, split and load data into dataloaders. You will need to download the dataset (if using) from [here](https://cocodataset.org/#home). Otherwise, simply construct a PyTorch Lightning DataModule to prepare your own dataset.
- coco_cat.py - Creates list of English-language labels corresponding to COCO label vector
- coco_dataset.py - Custom dataset that will load the COCO 2014 dataset and annotations
- cutmix.py - CutMix image regularization module

### q2l_labeller/models
- timm_backbone.py - A simple module that will download and initialize any desired backbone from the TIMM library.
- query2label.py - Includes my implementation of the overall architecture.

### q2l_labeller/pl_modules
- query2label_train_module.py - Contains training loop, evaluation methods and other Pytorch Lightning code.

### q2l_labeller/loss_modules
- simple_asymmetric_loss.py - Contains custom loss method from paper.
- partial_asymmetric_loss.py - Unused, but potentially promising loss module from Alibaba MIIL.

## Usage
### Training
To train this model with COCO 2014, simply run through the `1-simple-training-demo.ipynb` notebook. I've also provided the `2-full-demo.ipynb` notebook in case you'd like to see all the essential code in one place.

## Results
This model was able to get 90.4 mAP on COCO 2014, better than any of the CNN-backbone models used in the paper (but not as good as the vision transformer versions; you can try substituting these yourself if desired). Settings for achieving this are explained in the [blog post](https://medium.com/@curttigges/building-a-transformer-powered-sota-image-labeller-cfe25e6d69f1).
