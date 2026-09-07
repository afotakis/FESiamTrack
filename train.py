import logging 
import os
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import Optional
import torch.nn.init as init
from utils.callbacks import IncreaseSequenceLengthCallback
from utils.utils import *

logger = logging.getLogger(__name__)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.set_num_threads(1)
torch.backends.cudnn.benchmark = True

import os, subprocess, shutil
import pytorch_lightning as pl

class LaunchTensorBoardCallback(pl.Callback):
    """
    Starts a local TensorBoard server pointing to your logger directory.
    Use SSH port-forwarding if training on a remote box:
      ssh -N -L 6006:127.0.0.1:6006 <user>@<remote>
    """
    def __init__(self, port:int=6007, host:str="127.0.0.1"):
        super().__init__()
        self.port = int(port)
        self.host = host
        self.proc = None

    def on_fit_start(self, trainer, pl_module):
        if not trainer.logger:
            pl_module.print("No logger found; skip launching TensorBoard.")
            return
        if not shutil.which("tensorboard"):
            pl_module.print("⚠️ tensorboard not found. pip install tensorboard")
            return

        # Point TensorBoard to the parent of the versioned log dir
        # e.g., tb_runs/<name> (so you can compare versions)
        logdir = getattr(trainer.logger, "log_dir", None) or trainer.default_root_dir
        root = os.path.dirname(logdir) if os.path.isdir(logdir) else logdir

        try:
            self.proc = subprocess.Popen(
                ["tensorboard",
                 f"--logdir={root}",
                 f"--port={self.port}",
                 f"--host={self.host}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            pl_module.print(
                f"📈 TensorBoard live at http://{self.host}:{self.port}  (logdir: {root})\n"
                "If remote, tunnel with:  ssh -N -L 6006:127.0.0.1:6006 <user>@<remote>"
            )
        except Exception as e:
            pl_module.print(f"Failed to launch TensorBoard: {e}")

    def on_fit_end(self, trainer, pl_module):
        if self.proc:
            self.proc.terminate()
            self.proc = None





# put near your other callbacks
class PillowCompatCallback(pl.Callback):
    def on_fit_start(self, trainer, pl_module):
        # Create ANTIALIAS alias for Pillow >= 10
        try:
            from PIL import Image
            if not hasattr(Image, "ANTIALIAS") and hasattr(Image, "Resampling"):
                Image.ANTIALIAS = Image.Resampling.LANCZOS
        except Exception:
            pass


class SaveAtStep(pl.Callback):
    def __init__(self, target_step: int, filename: str):
        super().__init__()
        self.target_step = int(target_step)
        self.filename = filename
        self._done = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # global_step counts optimizer steps (respects grad accumulation)
        if self._done or trainer.global_step < self.target_step:
            return
        if trainer.global_step == self.target_step and trainer.is_global_zero:
            # choose a safe directory (logger dir if present; else default_root_dir)
            root = None
            if trainer.log_dir:  # PL >= 2.0
                root = trainer.log_dir
            elif trainer.logger and hasattr(trainer.logger, "log_dir"):
                root = trainer.logger.log_dir
            else:
                root = trainer.default_root_dir
            ckpt_dir = os.path.join(root, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            path = os.path.join(ckpt_dir, self.filename)
            trainer.save_checkpoint(path)
            print(f"[SaveAtStep] Saved checkpoint at step {trainer.global_step} -> {path}")
            self._done = True

class SaveEveryNSteps(pl.Callback):
    def __init__(self, n: int = 5000, filename_tmpl: str = "step_{step:06d}.ckpt"):
        super().__init__()
        self.n = int(n)
        self.filename_tmpl = filename_tmpl
        self._last_saved_step = -1

    def _ckpt_dir(self, trainer):
        if getattr(trainer, "log_dir", None):
            root = trainer.log_dir
        elif trainer.logger and hasattr(trainer.logger, "log_dir"):
            root = trainer.logger.log_dir
        else:
            root = trainer.default_root_dir
        ckpt_dir = os.path.join(root, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        return ckpt_dir

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step <= 0 or not trainer.is_global_zero:
            return
        if step % self.n != 0:
            return
        if step == self._last_saved_step:
            return
        ckpt_dir = self._ckpt_dir(trainer)
        path = os.path.join(ckpt_dir, self.filename_tmpl.format(step=step))
        trainer.save_checkpoint(path)
        print(f"[SaveEveryNSteps] Saved checkpoint at step {step} -> {path}")
        self._last_saved_step = step

class SaveEpochsAfterStep(pl.Callback):
    """
    Save a checkpoint at the end of every *training* epoch
    once global_step >= start_step. Filename includes epoch and step.
    """
    def __init__(self, start_step: int = 130_000,
                 filename_tmpl: str = "epoch_{epoch:03d}_step_{step:06d}.ckpt"):
        super().__init__()
        self.start_step = int(start_step)
        self.filename_tmpl = filename_tmpl

    def _ckpt_dir(self, trainer):
        # Prefer logger dir; fallback to default_root_dir
        if getattr(trainer, "log_dir", None):
            root = trainer.log_dir
        elif trainer.logger and hasattr(trainer.logger, "log_dir"):
            root = trainer.logger.log_dir
        else:
            root = trainer.default_root_dir
        ckpt_dir = os.path.join(root, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        return ckpt_dir

    def on_train_epoch_end(self, trainer, pl_module):
        # Only rank 0 saves
        if not trainer.is_global_zero:
            return

        step = trainer.global_step
        if step < self.start_step:
            return

        ckpt_dir = self._ckpt_dir(trainer)
        fname = self.filename_tmpl.format(
            epoch=trainer.current_epoch,
            step=step,
        )
        path = os.path.join(ckpt_dir, fname)
        trainer.save_checkpoint(path)
        pl_module.print(f"[SaveEpochsAfterStep] Saved {path}")


class ScaleLRAfterStep(pl.Callback):
    """
    From `start_step` onward, rescale LR by `factor` *after* the scheduler update,
    exactly once per optimizer step (safe with grad accumulation).
    """
    def __init__(self, start_step: int = 120_000, factor: float = 0.1):
        super().__init__()
        self.start_step = int(start_step)
        self.factor = float(factor)
        self._last_scaled_step = -1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Only act after real optimizer steps (global_step increments only then)
        step = trainer.global_step
        if step < self.start_step:
            return
        if step == self._last_scaled_step:
            return  # already scaled this optimizer step

        # Multiply current LR (set by OneCycle) by factor
        for opt in trainer.optimizers:
            for pg in opt.param_groups:
                if "lr" in pg:
                    pg["lr"] *= self.factor

        self._last_scaled_step = step
        if step == self.start_step:
            pl_module.print(f"[ScaleLRAfterStep] Applied x{self.factor} LR scale at step {step}.")

from pathlib import Path
def _tb_info_from_ckpt(ckpt_path: str):
    """
    Expect path like .../tb_runs/<run_name>/version_<int>/checkpoints/<file>.ckpt
    Returns: (save_dir, run_name, version_int)
    """
    p = Path(ckpt_path).resolve()
    if p.suffix != ".ckpt":
        raise ValueError(f"Not a .ckpt file: {ckpt_path}")
    if p.parent.name != "checkpoints":
        raise ValueError("Expected '.../version_*/checkpoints/<file>.ckpt'")
    version_dir = p.parent.parent.name              # e.g. 'version_0'
    run_name    = p.parent.parent.parent.name       # e.g. 'correlation3_unscaled_old'
    save_dir    = str(p.parent.parent.parent.parent)  # .../tb_runs
    if not version_dir.startswith("version_"):
        raise ValueError(f"Malformed version dir: {version_dir}")
    version_int = int(version_dir.split("_")[-1])
    return save_dir, run_name, version_int


@hydra.main(config_path="configs", config_name="train_defaults")
def train(cfg):
    pl.seed_everything(1234)

    propagate_keys(cfg)
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    # 0) Instantiate model (LightningModule) and datamodule
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    data_module = hydra.utils.instantiate(cfg.data)


    #Logging
    if cfg.logging:
        run_name = getattr(cfg, "run_name", None)
        if run_name is None:
            # try a sensible default
            run_name = getattr(getattr(cfg, "model", None), "name", None) or "run"
        
        training_logger = TensorBoardLogger(
            save_dir="tb_runs",      # nice, stable root outside hydra outputs
            name=run_name,           # tb_runs/<run_name>/version_x
            default_hp_metric=False,
            log_graph=True,
        )
        print("TensorBoard log dir:", training_logger.log_dir)
    else:
        training_logger = None
    

    # if cfg.logging:

    #         # make save_dir stable outside Hydra's changing run dir
    #     base = hydra.utils.get_original_cwd()
    #     save_dir = os.path.join(base, "tb_runs")
    #     run_name = getattr(cfg, "run_name", None) \
    #                    or getattr(getattr(cfg, "model", None), "name", None) \
    #                    or "run"
    #     version_int = None  # let PL create a fresh version

    #     training_logger = TensorBoardLogger(
    #         save_dir=save_dir,      # e.g. .../tb_runs
    #         name=run_name,          # e.g. correlation3_unscaled_old
    #         version=version_int,    # crucial to append to same version when resuming
    #         default_hp_metric=False,
    #         log_graph=True,
    #     )
    #     print("TensorBoard log dir:", training_logger.log_dir)
    # else:
    #     training_logger = None



    ckpt_cb = ModelCheckpoint(
        filename="{epoch}-{step}",
        save_last=True,          # also keep last.ckpt
        save_top_k=3,            # keep best 3 by val_loss
        monitor="loss/train_epoch",      # requires a val loop
        mode="min",
        every_n_epochs=1,        # or set every_n_train_steps=1000
        # dirpath=None -> Lightning uses the logger/Hydra run dir by default
    )
    save_140k_cb = SaveAtStep(target_step=140_000, filename="step_140k.ckpt")
    save_70k_cb = SaveAtStep(target_step=70_000, filename="step_70k.ckpt")
    save_139650k_cb = SaveAtStep(target_step=139_650, filename="step_139650k.ckpt")
    save_137738k_cb = SaveAtStep(target_step=137_738, filename="step_137738k.ckpt")

    
    # Training schedule + BN freeze callback
    callbacks = [
        PillowCompatCallback(),
        IncreaseSequenceLengthCallback(
            unroll_factor=cfg.unroll_factor, schedule=cfg.unroll_schedule
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        LaunchTensorBoardCallback(port=getattr(cfg, "tb_port", 6007)),
        SaveEveryNSteps(n=5000, filename_tmpl="step_{step:06d}.ckpt"),
        SaveEpochsAfterStep(start_step=120_000,
                        filename_tmpl="epoch_{epoch:03d}_step_{step:06d}.ckpt"),
        ckpt_cb,
    ]

    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer),
        devices=1,
        accelerator="gpu",
        callbacks=callbacks,
        logger=training_logger,
    )


    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    train()
