"""
tracking_experimentos.py - Integracion opcional con MLflow o WandB.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass


class BaseTracker:
    def log_config(self, config):
        return None

    def log_epoch(self, epoch: int, metrics: dict):
        return None

    def log_summary(self, metrics: dict):
        return None

    def log_artifact(self, path: str):
        return None

    def finish(self, status: str = "completed"):
        return None


class NullTracker(BaseTracker):
    pass


class MLflowTracker(BaseTracker):
    def __init__(self, project: str, run_name: str | None):
        import mlflow

        self.mlflow = mlflow
        mlflow.set_experiment(project)
        self.run = mlflow.start_run(run_name=run_name)

    def log_config(self, config):
        payload = asdict(config) if is_dataclass(config) else dict(config)
        self.mlflow.log_params(payload)

    def log_epoch(self, epoch: int, metrics: dict):
        self.mlflow.log_metrics(metrics, step=epoch)

    def log_summary(self, metrics: dict):
        self.mlflow.log_metrics(metrics)

    def log_artifact(self, path: str):
        self.mlflow.log_artifact(path)

    def finish(self, status: str = "completed"):
        self.mlflow.end_run(status=status.upper())


class WandBTracker(BaseTracker):
    def __init__(self, project: str, run_name: str | None):
        import wandb

        self.wandb = wandb
        self.run = wandb.init(project=project, name=run_name, reinit=True)

    def log_config(self, config):
        payload = asdict(config) if is_dataclass(config) else dict(config)
        self.wandb.config.update(payload, allow_val_change=True)

    def log_epoch(self, epoch: int, metrics: dict):
        payload = {"epoch": epoch}
        payload.update(metrics)
        self.wandb.log(payload)

    def log_summary(self, metrics: dict):
        self.wandb.log(metrics)

    def log_artifact(self, path: str):
        artifact = self.wandb.Artifact(name="rnn-artifacts", type="results")
        artifact.add_file(path)
        self.run.log_artifact(artifact)

    def finish(self, status: str = "completed"):
        code = 0 if status == "completed" else 1
        self.wandb.finish(exit_code=code)


def create_tracker(
    *,
    backend: str = "none",
    project: str = "rnn-sentimiento",
    run_name: str | None = None,
):
    backend = str(backend or "none").lower()
    if backend == "none":
        return NullTracker()
    if backend == "mlflow":
        try:
            return MLflowTracker(project=project, run_name=run_name)
        except ImportError:
            print("Aviso: mlflow no esta instalado. Se desactiva el tracking.")
            return NullTracker()
    if backend == "wandb":
        try:
            return WandBTracker(project=project, run_name=run_name)
        except ImportError:
            print("Aviso: wandb no esta instalado. Se desactiva el tracking.")
            return NullTracker()
    raise ValueError(f"Backend de tracking no soportado: {backend}")
