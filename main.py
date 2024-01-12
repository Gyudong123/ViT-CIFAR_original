import argparse

import comet_ml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
import warmup_scheduler
import numpy as np

from utils import get_model, get_dataset, get_experiment_name, get_criterion
from da import CutMix, MixUp

parser = argparse.ArgumentParser()
parser.add_argument("--api-key", default=None, help="API Key for Comet.ml")
parser.add_argument("--dataset", default="c100", type=str, help="[c10, c100, svhn]")
parser.add_argument("--num-classes", default=100, type=int)
parser.add_argument("--model-name", default="vit", help="[vit]", type=str)
parser.add_argument("--patch", default=8, type=int)
parser.add_argument("--batch-size", default=128, type=int)
parser.add_argument("--eval-batch-size", default=1024, type=int)
parser.add_argument("--lr", default=5e-4, type=float)
parser.add_argument("--min-lr", default=1e-5, type=float)
parser.add_argument("--beta1", default=0.9, type=float)
parser.add_argument("--beta2", default=0.999, type=float)
parser.add_argument("--off-benchmark", action="store_true")
parser.add_argument("--max-epochs", default=200, type=int)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--weight-decay", default=1e-4, type=float)
parser.add_argument("--warmup-epoch", default=10, type=int)
parser.add_argument("--precision", default=16, type=int)
parser.add_argument("--autoaugment",default=True, action="store_true")
parser.add_argument("--criterion", default="ce")
parser.add_argument("--label-smoothing", default=True, action="store_true")
parser.add_argument("--smoothing", default=0.1, type=float)
parser.add_argument("--rcpaste", action="store_true")
parser.add_argument("--cutmix", action="store_true")
parser.add_argument("--mixup", action="store_true")
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--head", default=12, type=int)
parser.add_argument("--num-layers", default=7, type=int)
parser.add_argument("--hidden", default=384, type=int)
parser.add_argument("--mlp-hidden", default=384, type=int)
parser.add_argument("--off-cls-token", action="store_true")
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--project-name", default="VisionTransformer")
args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
args.benchmark = True if not args.off_benchmark else False
args.gpus = torch.cuda.device_count()
args.num_workers = 4*args.gpus if args.gpus else 8
args.is_cls_token = True if not args.off_cls_token else False
args.min_lr = args.lr/100
if not args.gpus:
    args.precision=32

if args.mlp_hidden != args.hidden*4:
    print(f"[INFO] In original paper, mlp_hidden(CURRENT:{args.mlp_hidden}) is set to: {args.hidden*4}(={args.hidden}*4)")

train_ds, test_ds = get_dataset(args)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.eval_batch_size, num_workers=args.num_workers, pin_memory=True)

class EarlyStoppingAtEpoch50Callback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        # Check if the current epoch is 50 or more
        if trainer.current_epoch >= 50:
            # Get the validation accuracy from the logged metrics
            val_acc = trainer.logged_metrics.get('val_acc')
            # Check if validation accuracy is below 0.4
            if val_acc is not None and val_acc < 0.4:
                print(f"Stopping training at epoch {trainer.current_epoch} as validation accuracy is below 0.4")
                trainer.should_stop = True

class Net(pl.LightningModule):
    def __init__(self, hparams):
        super(Net, self).__init__()
        # self.hparams = hparams
        self.hparams.update(vars(hparams))
        self.model = get_model(hparams)
        self.criterion = get_criterion(args)
        if hparams.cutmix:
            self.cutmix = CutMix(hparams.size, beta=1.)
        if hparams.mixup:
            self.mixup = MixUp(alpha=1.)
        self.log_image_flag = hparams.api_key is None

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2), weight_decay=self.hparams.weight_decay)
        self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.min_lr)
        self.scheduler = warmup_scheduler.GradualWarmupScheduler(self.optimizer, multiplier=1., total_epoch=self.hparams.warmup_epoch, after_scheduler=self.base_scheduler)
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):
        img, label = batch
        if self.hparams.cutmix or self.hparams.mixup:
            if self.hparams.cutmix:
                img, label, rand_label, lambda_= self.cutmix((img, label))
            elif self.hparams.mixup:
                if np.random.rand() <= 0.8:
                    img, label, rand_label, lambda_ = self.mixup((img, label))
                else:
                    img, label, rand_label, lambda_ = img, label, torch.zeros_like(label), 1.
            out = self.model(img)
            loss = self.criterion(out, label)*lambda_ + self.criterion(out, rand_label)*(1.-lambda_)
        else:
            out = self(img)
            loss = self.criterion(out, label)

        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("loss", loss, on_step=False, on_epoch=True)
        self.log("acc", acc, on_step=False, on_epoch=True)
        # Check if loss is NaN and exit if true
        if torch.isnan(loss):
            exit('NaN loss encountered in validation, stopping training.')
        return loss

    def training_epoch_end(self, outputs):
        self.log("lr", self.optimizer.param_groups[0]["lr"], on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        img, label = batch
        out = self(img)
        loss = self.criterion(out, label)
        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        return loss


if __name__ == "__main__":
    experiment_name = get_experiment_name(args)
    print(experiment_name)
    if args.api_key:
        print("[INFO] Log with Comet.ml!")
        logger = pl.loggers.CometLogger(
            api_key=args.api_key,
            save_dir="logs",
            project_name=args.project_name,
            experiment_name=experiment_name
        )
        refresh_rate = 0
    else:
        print("[INFO] Log with CSV")
        logger = pl.loggers.CSVLogger(
            save_dir="logs",
            name=experiment_name
        )
        refresh_rate = 1
    net = Net(args)
    trainer = pl.Trainer(precision=args.precision,fast_dev_run=args.dry_run, gpus=args.gpus, benchmark=args.benchmark, logger=logger, max_epochs=args.max_epochs, weights_summary="full", progress_bar_refresh_rate=refresh_rate, callbacks=[EarlyStoppingAtEpoch50Callback()])
    trainer.fit(model=net, train_dataloader=train_dl, val_dataloaders=test_dl)
    if not args.dry_run:
        model_path = f"weights/{experiment_name}.pth"
        torch.save(net.state_dict(), model_path)
        if args.api_key:
            logger.experiment.log_asset(file_name=experiment_name, file_data=model_path)
