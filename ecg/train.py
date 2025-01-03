"""Training code."""

import random

import mlflow
import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAccuracy, BinaryConfusionMatrix, BinaryF1Score, BinaryPrecision, BinaryRecall

from ecg import DATA_DIR, PROJECT_ROOT, IMAGE_DIR
from ecg.dataset import ECGDataset
from ecg.loss import MSEBCELoss
from ecg.model import ConvAutoEncoder
from ecg.plotting import plot_reconstruction

class Trainer:
    """Model training setup."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer,
        scheduler,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
    ) -> None:
        """Deifne hyperparameters are utils for training."""
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.accuracy = BinaryAccuracy()
        self.confusion_matrix = BinaryConfusionMatrix()
        self.f1_score = BinaryF1Score()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()

        self.accuracy_history = []
        self.confusion_matrix_history = []
        self.f1_score_history = []
        self.precision_history = []
        self.recall_history = []

        self.train_batch_losses = []
        self.train_epoch_losses = []
        self.val_batch_losses = []
        self.val_epoch_losses = []

        self.train_batch_mse_losses = []
        self.train_epoch_mse_losses = []
        self.train_batch_bce_losses = []
        self.train_epoch_bce_losses = []

        self.val_batch_mse_losses = []
        self.val_epoch_mse_losses = []
        self.val_batch_bce_losses = []
        self.val_epoch_bce_losses = []

    def fit(self, epoch: int, device: str) -> None:
        """Train the model.

        Parameters
        ----------
        epoch: used for logging metrics
        """
        self.model.train()

        epoch_loss = 0
        epoch_mse_loss = 0
        epoch_bce_loss = 0

        for i, (X, target) in enumerate(self.train_dataloader):
            X, target = X.to(device), target.to(device)

            reconstructed, pred = self.model(X)

            loss = self.loss_fn(X, reconstructed, target, pred)
            mse_loss = self.loss_fn.mse_loss_weighted(X, reconstructed).item()
            bce_loss = self.loss_fn.bce_loss_weighted(target, pred).item()

            self.optimizer.zero_grad()  # The gradients are set to zero,
            loss.backward()  # the gradients are computed and stored.
            self.optimizer.step()  # .step() performs parameter update

            # Storing the losses in a list for plotting
            self.train_batch_losses.append(loss.item())
            epoch_loss += loss.item()
            self.train_batch_mse_losses.append(mse_loss)
            epoch_mse_loss += mse_loss
            self.train_batch_bce_losses.append(bce_loss)
            epoch_bce_loss += bce_loss

        # TODO: averaging this is wrong, should sum them average
        mean_epoch_loss = epoch_loss / i
        self.train_epoch_losses.append(mean_epoch_loss)
        mean_epoch_mse_loss = epoch_mse_loss / i
        self.train_epoch_mse_losses.append(mean_epoch_mse_loss)
        mean_epoch_bce_loss = epoch_bce_loss / i
        self.train_epoch_bce_losses.append(mean_epoch_bce_loss)

        metrics = {
            "train_loss": mean_epoch_loss,
            "train_mse_loss": mean_epoch_mse_loss,
            "train_bce_loss": mean_epoch_bce_loss
        }
        mlflow.log_metrics(metrics, epoch)

    def eval(self, epoch: int, device: str) -> None:
        """Evaluate the trained model.

        Parameters
        ----------
        epoch: used for logging metrics

        """
        self.model.eval()
        with torch.no_grad():
            epoch_loss = 0
            epoch_mse_loss = 0
            epoch_bce_loss = 0
            for j, (X, target) in enumerate(self.val_dataloader):
                X, target = X.to(device), target.to(device)

                reconstructed, pred = self.model(X)

                mse_loss = self.loss_fn.mse_loss_weighted(X, reconstructed).item()
                bce_loss = self.loss_fn.bce_loss_weighted(target, pred).item()
                loss = self.loss_fn(X, reconstructed, target, pred).item()

                self.val_batch_losses.append(loss)
                epoch_loss += loss
                self.val_batch_mse_losses.append(mse_loss)
                epoch_mse_loss += mse_loss
                self.val_batch_bce_losses.append(bce_loss)
                epoch_bce_loss += bce_loss

                pred = pred.squeeze()
                target = target.squeeze().int()

                self.accuracy.update(pred, target)
                self.confusion_matrix.update(pred, target)
                self.f1_score.update(pred, target)
                self.precision.update(pred, target)
                self.recall.update(pred, target)

            # TODO: averaging this is wrong, should sum them average
            mean_epoch_loss = epoch_loss / j
            mean_epoch_mse_loss = epoch_mse_loss / j
            mean_epoch_bce_loss = epoch_bce_loss / j
            self.val_epoch_losses.append(mean_epoch_loss)
            self.val_epoch_mse_losses.append(mean_epoch_mse_loss)
            self.val_epoch_bce_losses.append(mean_epoch_bce_loss)

            accuracy = self.accuracy.compute().item()
            confusion_matrix = self.confusion_matrix.compute()
            f1_score = self.f1_score.compute().item()
            precision = self.precision.compute().item()
            recall = self.recall.compute().item()
            self.accuracy_history.append(accuracy)
            self.confusion_matrix_history.append(confusion_matrix)
            self.f1_score_history.append(f1_score)
            self.precision_history.append(precision)
            self.recall_history.append(recall)
            
            metrics = {
                "val_loss": mean_epoch_loss,
                "val_mse_loss": mean_epoch_mse_loss,
                "val_bce_loss": mean_epoch_bce_loss,
                "accuracy": accuracy,
                "f1_score": f1_score,
                "precision": precision,
                "recall": recall
            }
            mlflow.log_metrics(metrics, epoch)

            self.accuracy.reset()
            self.confusion_matrix.reset()
            self.f1_score.reset()
            self.precision.reset()
            self.recall.reset()


def main(epochs: int) -> None:
    """Set up and training.

    Parameters
    ----------
    epochs: the number of epochs to train for.
    """
    abnormal_file = "ptbdb_abnormal.parquet"
    normal_file = "ptbdb_normal.parquet"

    abnormal_df = pl.read_parquet(DATA_DIR / abnormal_file)
    normal_df = pl.read_parquet(DATA_DIR / normal_file)

    df = pl.concat(
        [
            normal_df.with_columns(pl.Series("class", ["normal"] * normal_df.shape[0])),
            abnormal_df.with_columns(pl.Series("class", ["abnormal"] * abnormal_df.shape[0])),
        ],
        how="vertical",
    )

    dataset = ECGDataset(df.drop(["target", "class"]), df.select("target")) 
    
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # hyperparameters
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    train_val_test_splits = [0.8, 0.1, 0.1]
    input_dim = dataset[0][0].shape[0]
    hidden_dim = 128
    batch_size = 32
    kernel_size = 9
    stride = 1

    # optimizer
    lr = 1e-1
    weight_decay = 1e-8

    # scheduler
    patience = 2

    # loss
    mse_weight = 1000.0
    bce_weight = 1.0

    inputs = {
        "abnormal_data_file": DATA_DIR / abnormal_file,
        "normal_data_file": DATA_DIR / normal_file,
    }
    # mlflow.log_input(DATA_DIR / abnormal_file, context="abnormal data")
    # mlflow.log_input(DATA_DIR / normal_file, context="normal data")


    params = {
        "train_val_test_splits": train_val_test_splits,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "hidden_dims": hidden_dim,
        "kernel_size": kernel_size,
        "stride": stride,
        "lr": lr,
        "weight_decay": weight_decay,
        "patience": 2,
        "mse_weight": mse_weight,
        "bce_weight": bce_weight,
    }
    mlflow.log_params(params)
    mlflow.set_tag("Architecture", "Convolutional Autoencoder")
    mlflow.set_tag("Hidden Units", 3)
    mlflow.set_tag("Classifier", "1 Layer MLP")

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, train_val_test_splits)
    model = ConvAutoEncoder(input_dim, hidden_dim, kernel_size, stride)
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=patience)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    gen = torch.Generator()
    gen.manual_seed(0)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=gen,
        # num_workers=4,
        # persistent_workers=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=gen,
        # num_workers=4,
        # persistent_workers=True
    )

    loss_fn = MSEBCELoss(mse_weight=mse_weight, bce_weight=bce_weight)

    trainer = Trainer(model, loss_fn, optimizer, scheduler, train_dataloader, val_dataloader)

    for epoch in range(epochs):
        trainer.fit(epoch, device)
        trainer.eval(epoch, device)

        val_epoch_loss = trainer.val_epoch_losses[-1]
        train_mse_loss = trainer.train_epoch_mse_losses[-1]
        train_bce_loss = trainer.train_epoch_bce_losses[-1]
        val_mse_loss = trainer.val_epoch_mse_losses[-1]
        val_bce_loss = trainer.val_epoch_bce_losses[-1]
        scheduler.step(val_epoch_loss)  # update the learning rate if not learning
        
        lr = scheduler.get_last_lr()[0]
        print(
            f"{epoch}/{epochs} (lr={lr:0.6f}) - train loss: ({train_mse_loss:0.6f}, {train_bce_loss:0.6f}) | val loss: ({val_mse_loss:0.6f}, {val_bce_loss:0.6f})"
        )
        mlflow.log_metric("lr", lr, step=epoch)

    
    ####################
    # Save Model
    ####################
    model = model.to("cpu")
    example = train_dataset[0][0][None, :]
    output = model(example)
    autoencoder_signature = mlflow.models.infer_signature(
        model_input=example.numpy(),
        model_output=output[0].detach().numpy()
    )
    classifier_signature = mlflow.models.infer_signature(
        model_input=example.numpy(),
        model_output=output[1].detach().numpy()
    )

    mlflow.pytorch.log_model(
        model, 
        "Convolutional AutoEncoder",
        signature=autoencoder_signature,
        pip_requirements=(PROJECT_ROOT / "pyproject.toml").as_posix()
    )
    mlflow.pytorch.log_model(
        model, 
        "Classifier",
        signature=classifier_signature,
        pip_requirements=(PROJECT_ROOT / "pyproject.toml").as_posix()
    )

    ####################
    # Plotting
    ####################
    # Plot the first batch of validation data
    X, target = next(iter(val_dataloader))
    reconstructed, pred = model(X)
    
    for idx in range(len(X)):

        target_class = "normal" if target[idx].squeeze() == 0.0 else "abnormal"
        pred_class = "normal" if pred[idx].squeeze() == 0.0 else "abnormal"
        plot = plot_reconstruction(
            pl.DataFrame(X[idx].numpy()), 
            pl.DataFrame(reconstructed[idx].detach().numpy()), 
            target_class, 
            pred_class
        )
        
        filename = (IMAGE_DIR / f"reconstruction_{str(idx).zfill(3)}.png").as_posix()
        plot.save(filename)
        mlflow.log_artifact(filename, "images")



if __name__ == "__main__":
    from pathlib import Path
    import mlflow
    
    epochs = 20

    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    tracking_uri = mlflow.get_tracking_uri()
    print(f"MLFLow Server: {tracking_uri}")
    
    mlflow.set_experiment("ECG")

    with mlflow.start_run() as run:
        artifact_uri = Path(run.info.artifact_uri)
        experiment_id = artifact_uri.parent.parent.stem
        print(f"Experiment: {run.info.run_name}")

        run_uri = tracking_uri + "/#/experiments/" + experiment_id + "/runs/" + run.info.run_id
        print(f"🏃 View run [{run.info.run_name}] {run_uri}")
        main(epochs)
