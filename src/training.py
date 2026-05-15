import math
from pathlib import Path
import shutil

from tqdm import tqdm
import torch, torch.nn as nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from src.config import (
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    WARMUP_STEPS,
    GRAD_CLIP,
    FREEZE_CNN_EPOCHS,
    CHECKPOINT_DIR,
    LOG_DIR,
    DEVICE,
    LABEL_SMOOTHING,
    VAL_METRICS_EVERY_N_EPOCHS,
    SAVE_RESUME_EVERY_EPOCH,
    RESUME_CHECKPOINT_FILENAME,
)
from src.balancing import LabelSmoothingCrossEntropy
from src.utils import set_seed

set_seed()


def _backup_to_drive(local_path):
    """Copy checkpoint to Google Drive if mounted."""
    try:
        drive_root = Path("/content/drive/MyDrive")
        if not drive_root.exists():
            return
        # Create a backup folder next to your project
        backup_dir = drive_root / "ImageCaptioningProject" / "image-captioning" / "outputs" / "checkpoints_backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, backup_dir / Path(local_path).name)
        print(f"[OK] Backed up to Drive: {Path(local_path).name}")
    except Exception as e:
        print(f"[WARN] Drive backup failed: {e}")


def _training_config_snapshot():
    import src.config as C

    keys = (
        "NUM_EPOCHS",
        "BATCH_SIZE",
        "LEARNING_RATE",
        "WEIGHT_DECAY",
        "WARMUP_STEPS",
        "GRAD_CLIP",
        "FREEZE_CNN_EPOCHS",
        "LABEL_SMOOTHING",
        "MIN_WORD_FREQ",
        "SPLIT_SEED",
        "TRAIN_FRAC",
        "VAL_FRAC",
        "TEST_FRAC",
        "SEED",
        "D_MODEL",
        "NUM_DECODER_LAYERS",
        "NUM_HEADS",
        "DIM_FF",
        "DROPOUT",
        "VAL_METRICS_EVERY_N_EPOCHS",
    )
    return {k: getattr(C, k) for k in keys if hasattr(C, k)}


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr=LEARNING_RATE, min_lr=0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            p = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * p))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        return {"current_step": self.current_step}

    def load_state_dict(self, state):
        self.current_step = int(state["current_step"])


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        vocab,
        checkpoint_name="best_model.pt",
        val_references=None,
    ):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab = vocab
        self.val_references = val_references
        self.pad_idx = vocab.word2idx['<<PAD>']
        self.checkpoint_path = CHECKPOINT_DIR / checkpoint_name
        self.criterion = LabelSmoothingCrossEntropy(self.pad_idx, LABEL_SMOOTHING)
        self.writer = SummaryWriter(LOG_DIR)
        cnn_params = [p for n, p in model.named_parameters() if n.startswith('encoder') and p.requires_grad]
        dec_params = [p for n, p in model.named_parameters() if not n.startswith('encoder') and p.requires_grad]
        self.optimizer = AdamW(
            [{'params': dec_params, 'lr': LEARNING_RATE},
             {'params': cnn_params, 'lr': LEARNING_RATE * 0.1}],
            betas=(0.9, 0.98), eps=1e-9, weight_decay=WEIGHT_DECAY
        )
        total_steps = NUM_EPOCHS * len(train_loader)
        self.scheduler = WarmupCosineScheduler(self.optimizer, WARMUP_STEPS, total_steps)
        self.best_val_loss = float('inf')
        self.global_step = 0

    def _set_cnn_grad(self, freeze):
        for p in self.model.encoder.resnet.parameters():
            p.requires_grad = not freeze

    def train_epoch(self, epoch):
        self.model.train()
        freeze = epoch < FREEZE_CNN_EPOCHS
        self._set_cnn_grad(freeze)
        if freeze:
            print(f'[Epoch {epoch+1}] CNN frozen.')
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f'Train {epoch+1}/{NUM_EPOCHS}')
        for images, captions, _ in pbar:
            images, captions = images.to(DEVICE), captions.to(DEVICE)
            tgt_input = captions[:, :-1]
            tgt_output = captions[:, 1:]
            tgt_mask = self.model.decoder.make_mask(tgt_input.size(1)).to(DEVICE)
            tgt_pad_mask = (tgt_input == self.pad_idx).to(DEVICE)
            self.optimizer.zero_grad()
            outputs = self.model(images, tgt_input, tgt_mask, tgt_pad_mask)
            outputs = outputs.reshape(-1, outputs.size(-1))
            tgt_output = tgt_output.reshape(-1)
            loss = self.criterion(outputs, tgt_output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            self.global_step += 1
            pbar.set_postfix(loss=f'{loss.item():.4f}', lr=f'{self.scheduler.get_lr():.2e}')
            self.writer.add_scalar('Train/loss', loss.item(), self.global_step)
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        for images, captions, _ in tqdm(self.val_loader, desc='Validate'):
            images, captions = images.to(DEVICE), captions.to(DEVICE)
            tgt_input = captions[:, :-1]
            tgt_output = captions[:, 1:]
            tgt_mask = self.model.decoder.make_mask(tgt_input.size(1)).to(DEVICE)
            tgt_pad_mask = (tgt_input == self.pad_idx).to(DEVICE)
            outputs = self.model(images, tgt_input, tgt_mask, tgt_pad_mask)
            outputs = outputs.reshape(-1, outputs.size(-1))
            tgt_output = tgt_output.reshape(-1)
            total_loss += self.criterion(outputs, tgt_output).item()
        avg = total_loss / len(self.val_loader)
        self.writer.add_scalar('Val/loss', avg, epoch)
        self.writer.add_scalar('Val/ppl', math.exp(avg), epoch)
        print(f'[Epoch {epoch+1}] Val Loss={avg:.4f} | PPL={math.exp(avg):.2f}')
        if avg < self.best_val_loss:
            self.best_val_loss = avg
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg,
                    'vocab': self.vocab,
                },
                self.checkpoint_path,
            )
            print('[OK] New best checkpoint saved.')
            _backup_to_drive(self.checkpoint_path)   # <-- backup best model
        return avg

    @torch.no_grad()
    def val_generation_metrics(self, epoch):
        if self.val_references is None or VAL_METRICS_EVERY_N_EPOCHS <= 0:
            return
        if (epoch + 1) % VAL_METRICS_EVERY_N_EPOCHS != 0:
            return
        from src.evaluation import Evaluator

        self.model.eval()
        ev = Evaluator(self.model, self.vocab)
        metrics = ev.evaluate(
            self.val_loader,
            self.val_references,
            desc=f"Val metrics {epoch + 1}/{NUM_EPOCHS}",
            verbose=False,
            return_predictions=False,
        )
        for k in ("bleu1", "bleu4", "meteor"):
            self.writer.add_scalar(f"Val/{k}", float(metrics[k]), epoch)
        print(
            f'[Epoch {epoch + 1}] Val BLEU-1={metrics["bleu1"]:.4f} | '
            f'BLEU-4={metrics["bleu4"]:.4f} | METEOR={metrics["meteor"]:.4f}'
        )

    def _save_resume_checkpoint(self, epoch):
        if not SAVE_RESUME_EVERY_EPOCH:
            return
        path = CHECKPOINT_DIR / RESUME_CHECKPOINT_FILENAME
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "best_val_loss": self.best_val_loss,
                "global_step": self.global_step,
                "vocab": self.vocab,
                "training_config": _training_config_snapshot(),
            },
            path,
        )
        print(f"[OK] Resume checkpoint saved: {path}")
        _backup_to_drive(path)  # <-- backup resume checkpoint

    def fit(self, resume_path=None):
        print(f"[INFO] Training on {DEVICE}")
        start_epoch = 0
        if resume_path is not None:
            p = Path(resume_path)
            if not p.is_file():
                raise FileNotFoundError(f"Resume checkpoint not found: {p}")
            ckpt = torch.load(p, map_location=DEVICE, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
            self.global_step = ckpt.get("global_step", 0)
            start_epoch = int(ckpt["epoch"]) + 1
            snap = ckpt.get("training_config")
            if snap:
                print(f"[INFO] Resuming from epoch {start_epoch}/{NUM_EPOCHS}; checkpoint config: {snap}")
            else:
                print(f"[INFO] Resuming from epoch {start_epoch}/{NUM_EPOCHS}")
        if start_epoch >= NUM_EPOCHS:
            print(
                f"[WARN] Checkpoint is already at or past NUM_EPOCHS ({NUM_EPOCHS}); "
                "no further training steps will run."
            )
            self.writer.close()
            return
        for epoch in range(start_epoch, NUM_EPOCHS):
            tl = self.train_epoch(epoch)
            vl = self.validate(epoch)
            self.val_generation_metrics(epoch)
            print(f"[Epoch {epoch + 1}] Train={tl:.4f} | Val={vl:.4f}\n")
            self._save_resume_checkpoint(epoch)
        self.writer.close()