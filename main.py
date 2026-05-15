import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from src.utils import set_seed
from src.config import IMAGE_DIR, CAPTION_FILE, BATCH_SIZE, CHECKPOINT_DIR, DEVICE
from src.preprocessing import build_data_loaders
from src.architecture import ImageCaptioningModel
from src.training import Trainer
from src.evaluation import Evaluator
import torch

def main():
    parser = argparse.ArgumentParser(description="Train image captioning model")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to full resume file (e.g. outputs/checkpoints/last.pt) for Colab/PC handoff",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip held-out test metrics after training (e.g. if best_model.pt missing)",
    )
    args = parser.parse_args()
    set_seed()
    print('[INFO] Building data loaders (disjoint train / val / test by image)...')
    train_loader, val_loader, test_loader, vocab, val_references, test_references = build_data_loaders(
        IMAGE_DIR, CAPTION_FILE, BATCH_SIZE
    )
    print('[INFO] Initializing model...')
    model = ImageCaptioningModel(vocab.n_words, 512, 6, 8, 2048, 0.5, 14)
    print(f'[INFO] Parameters: {sum(p.numel() for p in model.parameters()):,}')
    print('[INFO] Starting training...')
    Trainer(model, train_loader, val_loader, vocab, val_references=val_references).fit(resume_path=args.resume)
    print('[OK] Training complete.')
    if args.skip_test:
        print('[INFO] Skipping test evaluation (--skip-test).')
        return
    best_path = CHECKPOINT_DIR / 'best_model.pt'
    if not best_path.is_file():
        print(f'[WARN] No checkpoint at {best_path}; skip test evaluation.')
        return
    print('[INFO] Loading best checkpoint for held-out test evaluation...')
    ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    Evaluator(model, vocab).evaluate(test_loader, test_references)
    print('[OK] Test evaluation complete.')

if __name__ == '__main__':
    main()
