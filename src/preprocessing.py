import json, re, pickle
from pathlib import Path
from functools import partial
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from src.config import (
    IMAGE_DIR,
    CAPTION_FILE,
    PROCESSED_DIR,
    MIN_WORD_FREQ,
    CAPTION_MAX_LEN,
    BATCH_SIZE,
    SPLIT_SEED,
    TRAIN_FRAC,
    VAL_FRAC,
    TEST_FRAC,
)

def clean_caption(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text)

class Vocabulary:
    def __init__(self):
        self.word2idx = {'<<PAD>':0, '<SOS>':1, '<EOS>':2, '<UNK>':3}
        self.idx2word = {v:k for k,v in self.word2idx.items()}
        self.n_words = 4

    def build_vocab(self, captions, min_freq=MIN_WORD_FREQ):
        cnt = Counter(w for cap in captions for w in cap.split())
        for word, c in cnt.items():
            if c >= min_freq and word not in self.word2idx:
                self.word2idx[word] = self.n_words
                self.idx2word[self.n_words] = word
                self.n_words += 1

    def numericalize(self, text):
        return [self.word2idx.get(w, self.word2idx['<UNK>']) for w in text.split()]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"w2i":self.word2idx, "i2w":self.idx2word, "n":self.n_words}, f)

    @classmethod
    def load(cls, path):
        o = cls()
        with open(path, "rb") as f:
            d = pickle.load(f)
        o.word2idx, o.idx2word, o.n_words = d["w2i"], d["i2w"], d["n"]
        return o

class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, captions_dict, vocab, transform=None):
        self.image_dir = Path(image_dir)
        self.vocab = vocab
        self.transform = transform
        self.samples = [(img, c) for img, caps in captions_dict.items() for c in caps]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_name, caption = self.samples[idx]
        img = Image.open(self.image_dir / img_name).convert("RGB")
        if self.transform: img = self.transform(img)
        caption = clean_caption(caption)
        tokens = [self.vocab.word2idx['<SOS>']] + self.vocab.numericalize(caption) + [self.vocab.word2idx['<EOS>']]
        if len(tokens) > CAPTION_MAX_LEN:
            tokens = tokens[:CAPTION_MAX_LEN]
            tokens[-1] = self.vocab.word2idx['<EOS>']
        return img, torch.tensor(tokens, dtype=torch.long), img_name

def collate_fn(batch, pad_idx=0):
    imgs = torch.stack([b[0] for b in batch])
    caps = [b[1] for b in batch]
    names = [b[2] for b in batch]
    max_len = max(len(c) for c in caps)
    padded = torch.full((len(caps), max_len), pad_idx, dtype=torch.long)
    for i, c in enumerate(caps): padded[i, :len(c)] = c
    return imgs, padded, names

def read_captions(path):
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if "	" in line: img, cap = line.split("	", 1)
            else:
                parts = line.split(",", 1)
                if len(parts) != 2: continue
                img, cap = parts
            img = img.split("#")[0]
            d.setdefault(img, []).append(cap)
    return d

def subset_caps_dict(caps_dict, image_keys):
    return {k: caps_dict[k] for k in image_keys if k in caps_dict}

def split_caps_dict_by_image(caps_dict, train_f=TRAIN_FRAC, val_f=VAL_FRAC, test_f=TEST_FRAC, seed=SPLIT_SEED):
    if abs(train_f + val_f + test_f - 1.0) > 1e-6:
        raise ValueError("TRAIN_FRAC + VAL_FRAC + TEST_FRAC must sum to 1.0")
    keys = np.array(sorted(caps_dict.keys()))
    if len(keys) < 3:
        raise ValueError(f"Need at least 3 images for train/val/test split; got {len(keys)}.")
    train_val_keys, test_keys = train_test_split(
        keys, test_size=test_f, random_state=seed, shuffle=True
    )
    val_ratio_in_train_val = val_f / (train_f + val_f)
    train_keys, val_keys = train_test_split(
        train_val_keys, test_size=val_ratio_in_train_val, random_state=seed, shuffle=True
    )
    train_d = subset_caps_dict(caps_dict, train_keys)
    val_d = subset_caps_dict(caps_dict, val_keys)
    test_d = subset_caps_dict(caps_dict, test_keys)
    return train_d, val_d, test_d

def references_dict_for_eval(caps_dict):
    """Multiple references per image (tokenization happens in Evaluator)."""
    return {img: [clean_caption(c) for c in caps] for img, caps in caps_dict.items()}

def build_data_loaders(
    image_dir=IMAGE_DIR,
    caption_file=CAPTION_FILE,
    batch_size=BATCH_SIZE,
    num_workers=0,
    split_seed=SPLIT_SEED,
):
    from src.augmentation import get_train_transforms, get_val_transforms
    caps_dict = read_captions(caption_file)
    train_d, val_d, test_d = split_caps_dict_by_image(
        caps_dict, TRAIN_FRAC, VAL_FRAC, TEST_FRAC, split_seed
    )
    manifest = {
        "split_seed": split_seed,
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "test_frac": TEST_FRAC,
        "n_images": len(caps_dict),
        "n_train_images": len(train_d),
        "n_val_images": len(val_d),
        "n_test_images": len(test_d),
        "train_images": sorted(train_d.keys()),
        "val_images": sorted(val_d.keys()),
        "test_images": sorted(test_d.keys()),
    }
    with open(PROCESSED_DIR / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    train_caps = [clean_caption(c) for caps in train_d.values() for c in caps]
    vocab = Vocabulary()
    vocab.build_vocab(train_caps, MIN_WORD_FREQ)
    vocab.save(PROCESSED_DIR / "vocab.pkl")
    print(
        f"[INFO] Split (by image): train={len(train_d)} val={len(val_d)} test={len(test_d)} | "
        f"vocab (train captions only): {vocab.n_words} words"
    )
    train_ds = Flickr8kDataset(image_dir, train_d, vocab, get_train_transforms())
    val_ds = Flickr8kDataset(image_dir, val_d, vocab, get_val_transforms())
    test_ds = Flickr8kDataset(image_dir, test_d, vocab, get_val_transforms())
    pad_idx = vocab.word2idx['<<PAD>']
    collate = partial(collate_fn, pad_idx=pad_idx)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate
    )
    val_references = references_dict_for_eval(val_d)
    test_references = references_dict_for_eval(test_d)
    return train_loader, val_loader, test_loader, vocab, val_references, test_references
