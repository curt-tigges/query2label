import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import precision
import torchmetrics.functional as tf

from q2l_labeller.data.cutmix import CutMixCriterion
from q2l_labeller.loss_modules.simple_asymmetric_loss import AsymmetricLoss
from q2l_labeller.models.query2label import Query2Label


class Query2LabelTrainModule(pl.LightningModule):
    def __init__(
        self,
        data,
        backbone_desc,
        conv_out_dim,
        hidden_dim,
        num_encoders,
        num_decoders,
        num_heads,
        batch_size,
        image_dim,
        learning_rate,
        momentum,
        weight_decay,
        n_classes,
        thresh=0.5,
        use_cutmix=False,
        use_pos_encoding=False,
        loss="BCE",
    ):
        super().__init__()

        # Key parameters
        self.save_hyperparameters(ignore=["model", "data"])
        self.data = data
        self.model = Query2Label(
            model=backbone_desc,
            conv_out=conv_out_dim,
            num_classes=n_classes,
            hidden_dim=hidden_dim,
            nheads=num_heads,
            encoder_layers=num_encoders,
            decoder_layers=num_decoders,
            use_pos_encoding=use_pos_encoding,
        )
        if loss == "BCE":
            self.base_criterion = nn.BCEWithLogitsLoss()
        elif loss == "ASL":
            self.base_criterion = AsymmetricLoss(gamma_neg=1, gamma_pos=0)

        self.criterion = CutMixCriterion(self.base_criterion)

    def forward(self, x):
        x = self.model(x)
        return x

    def evaluate(self, batch, stage=None):
        x, y = batch
        y_hat = self(x)
        loss = self.base_criterion(y_hat, y.type(torch.float))

        rmap = tf.retrieval_average_precision(y_hat, y.type(torch.int))

        category_prec = precision(
            y_hat,
            y.type(torch.int),
            average="macro",
            num_classes=self.hparams.n_classes,
            threshold=self.hparams.thresh,
            multiclass=False,
        )
        category_recall = tf.recall(
            y_hat,
            y.type(torch.int),
            average="macro",
            num_classes=self.hparams.n_classes,
            threshold=self.hparams.thresh,
            multiclass=False,
        )
        category_f1 = tf.f1_score(
            y_hat,
            y.type(torch.int),
            average="macro",
            num_classes=self.hparams.n_classes,
            threshold=self.hparams.thresh,
            multiclass=False,
        )

        overall_prec = precision(
            y_hat, y.type(torch.int), threshold=self.hparams.thresh, multiclass=False
        )
        overall_recall = tf.recall(
            y_hat, y.type(torch.int), threshold=self.hparams.thresh, multiclass=False
        )
        overall_f1 = tf.f1_score(
            y_hat, y.type(torch.int), threshold=self.hparams.thresh, multiclass=False
        )

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_rmap", rmap, prog_bar=True, on_step=False, on_epoch=True)

            self.log(f"{stage}_cat_prec", category_prec, prog_bar=True)
            self.log(f"{stage}_cat_recall", category_recall, prog_bar=True)
            self.log(f"{stage}_cat_f1", category_f1, prog_bar=True)

            self.log(f"{stage}_ovr_prec", overall_prec, prog_bar=True)
            self.log(f"{stage}_ovr_recall", overall_recall, prog_bar=True)
            self.log(f"{stage}_ovr_f1", overall_f1, prog_bar=True)

            # log prediction examples to wandb
            """
            pred = self.model(x)
            pred_keys = pred[0].sigmoid().tolist()
            pred_keys = [0 if p < self.hparams.thresh else 1 for p in pred_keys]


            mapper = cc.COCOCategorizer()
            pred_lbl = mapper.get_labels(pred_keys)
            
            try:
                self.logger.experiment.log({"val_pred_examples": [wandb.Image(x[0], caption=pred_lbl)]})
            except AttributeError:
                pass
            """

    def training_step(self, batch, batch_idx):
        if self.hparams.use_cutmix:
            x, y = batch
            y_hat = self(x)
            # y1, y2, lam = y
            loss = self.criterion(y_hat, y)

        else:
            x, y = batch
            y_hat = self(x)
            loss = self.base_criterion(y_hat, y.type(torch.float))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.hparams.weight_decay,
        )

        lr_scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                self.hparams.learning_rate,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=len(self.data.train_dataloader()),
                anneal_strategy="cos",
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
        # return optimizer
