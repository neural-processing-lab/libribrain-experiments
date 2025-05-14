from torch import nn
from torchmetrics import Accuracy, Precision, Recall
from pytorch_lightning import LightningModule
from torchmetrics import F1Score
from .utils import modules_from_config, optimizer_from_config, loss_fn_from_config


class ClassificationModule(LightningModule):
    def __init__(self, model_config: list[tuple[str, dict]], n_classes: int, optimizer_config: dict, loss_config: dict):
        super().__init__()
        self.save_hyperparameters()
        self.modules_list = nn.ModuleList()
        self.modules_list.extend(modules_from_config(model_config))
        self.loss_fn = loss_fn_from_config(loss_config)
        self.accuracy = Accuracy(num_classes=n_classes, task="multiclass")
        self.balanced_accuracy = Accuracy(
            num_classes=n_classes, task="multiclass", average="macro")
        self.f1_micro = F1Score(num_classes=n_classes,
                                task="multiclass", average="micro")
        self.f1_macro = F1Score(num_classes=n_classes,
                                task="multiclass", average="macro")
        self.precision_micro = Precision(
            num_classes=n_classes, task="multiclass", average="micro")
        self.precision_macro = Precision(
            num_classes=n_classes, task="multiclass", average="macro")

        self.optimizer_config = optimizer_config

        self.binary_accuracy = Accuracy(task="binary")
        self.binary_precision = Precision(task="binary")
        self.binary_recall = Recall(task="binary")
        self.binary_f1 = F1Score(task="binary")

    def forward(self, x):
        for module in self.modules_list:
            x = module(x)
        return x

    def configure_optimizers(self):
        return optimizer_from_config(self.parameters(), self.optimizer_config)

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.accuracy(y_hat, y)
        f1_micro = self.f1_micro(y_hat, y)
        f1_macro = self.f1_macro(y_hat, y)
        bal_acc = self.balanced_accuracy(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        self.log('train_f1_micro', f1_micro)
        self.log('train_f1_macro', f1_macro)
        self.log('train_precision_micro', self.precision_micro(y_hat, y))
        self.log('train_precision_macro', self.precision_macro(y_hat, y))
        self.log('train_bal_acc', bal_acc)
        for class_idx in range(y_hat.shape[1]):
            y_binary = (y == class_idx).int()
            y_hat_binary = y_hat.argmax(dim=1) == class_idx
            binary_acc = self.binary_accuracy(y_hat_binary, y_binary)
            self.log(f'train_acc_class_{class_idx}', binary_acc)

            monitored_class_prediction_rate = y_hat_binary.float().mean()
            self.log(f'train_prediction_rate_class_{class_idx}',
                     monitored_class_prediction_rate)
            monitored_class_recall = self.binary_recall(y_hat_binary, y_binary)
            self.log(f'train_recall_class_{class_idx}', monitored_class_recall)
            monitored_class_precision = self.binary_precision(
                y_hat_binary, y_binary)
            self.log(f'train_precision_class_{class_idx}',
                     monitored_class_precision)
            y_hat_probs = nn.functional.softmax(y_hat, dim=1)
            monitored_class_mean_probs = y_hat_probs[:, class_idx].mean(
            )
            self.log(f'train_mean_probability_class_{class_idx}',
                     monitored_class_mean_probs)
            self.log(f'train_f1_class_{class_idx}', self.binary_f1(
                y_hat_binary, y_binary))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.accuracy(y_hat, y)
        f1_micro = self.f1_micro(y_hat, y)
        f1_macro = self.f1_macro(y_hat, y)
        bal_acc = self.balanced_accuracy(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        self.log('val_f1_micro', f1_micro)
        self.log('val_f1_macro', f1_macro)
        self.log('val_precision_micro', self.precision_micro(y_hat, y))
        self.log('val_precision_macro', self.precision_macro(y_hat, y))
        self.log('val_bal_acc', bal_acc)
        for class_idx in range(y_hat.shape[1]):
            y_binary = (y == class_idx).int()
            y_hat_binary = y_hat.argmax(dim=1) == class_idx
            binary_acc = self.binary_accuracy(y_hat_binary, y_binary)
            self.log(f'val_acc_class_{class_idx}', binary_acc)
            monitored_class_prediction_rate = y_hat_binary.float().mean()
            self.log(f'val_prediction_rate_class_{class_idx}',
                     monitored_class_prediction_rate)
            monitored_class_recall = self.binary_recall(y_hat_binary, y_binary)
            self.log(f'val_recall_class_{class_idx}', monitored_class_recall)
            monitored_class_precision = self.binary_precision(
                y_hat_binary, y_binary)
            self.log(f'val_precision_class_{class_idx}',
                     monitored_class_precision)
            self.log(f'val_f1_class_{class_idx}',
                     self.binary_f1(y_hat_binary, y_binary))
            y_hat_probs = nn.functional.softmax(y_hat, dim=1)
            monitored_class_mean_probs = y_hat_probs[:, class_idx].mean(
            )
            self.log(f'val_mean_probability_class_{class_idx}',
                     monitored_class_mean_probs)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.accuracy(y_hat, y)
        f1_micro = self.f1_micro(y_hat, y)
        f1_macro = self.f1_macro(y_hat, y)
        bal_acc = self.balanced_accuracy(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        self.log('test_f1_micro', f1_micro)
        self.log('test_f1_macro', f1_macro)
        self.log('test_precision_micro', self.precision_micro(y_hat, y))
        self.log('test_precision_macro', self.precision_macro(y_hat, y))
        self.log('test_bal_acc', bal_acc)
        for class_idx in range(y_hat.shape[1]):
            y_binary = (y == class_idx).int()
            y_hat_binary = y_hat.argmax(dim=1) == class_idx
            binary_acc = self.binary_accuracy(y_hat_binary, y_binary)
            self.log(f'test_acc_class_{class_idx}', binary_acc)
            monitored_class_prediction_rate = y_hat_binary.float().mean()
            self.log(f'test_prediction_rate_class_{class_idx}',
                     monitored_class_prediction_rate)
            monitored_class_recall = self.binary_recall(y_hat_binary, y_binary)
            self.log(f'test_recall_class_{class_idx}', monitored_class_recall)
            monitored_class_precision = self.binary_precision(
                y_hat_binary, y_binary)
            self.log(f'test_precision_class_{class_idx}',
                     monitored_class_precision)
            y_hat_probs = nn.functional.softmax(y_hat, dim=1)
            monitored_class_mean_probs = y_hat_probs[:, class_idx].mean(
            )
            self.log(f'test_mean_probability_class_{class_idx}',
                     monitored_class_mean_probs)

            self.log(f'test_f1_class_{class_idx}',
                     self.binary_f1(y_hat_binary, y_binary))
        return loss
