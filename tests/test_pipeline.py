import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
from src.utils import set_seed
from src.config import DEVICE
from src.architecture import ImageCaptioningModel
from src.preprocessing import Vocabulary

def test_overfit_one_batch():
    set_seed(42)
    vocab = Vocabulary()
    vocab.word2idx = {'<<PAD>':0,'<SOS>':1,'<EOS>':2,'<UNK>':3,'dog':4,'runs':5}
    vocab.idx2word = {v:k for k,v in vocab.word2idx.items()}; vocab.n_words = len(vocab.word2idx)
    model = ImageCaptioningModel(vocab.n_words, d_model=64, num_layers=2, num_heads=2, dim_ff=128, dropout=0.0).to(DEVICE)
    dummy_img = torch.randn(2,3,224,224).to(DEVICE)
    dummy_cap = torch.randint(0,vocab.n_words,(2,8)).to(DEVICE)
    tgt_in = dummy_cap[:, :-1]
    tgt_out = dummy_cap[:, 1:]
    mask = model.decoder.make_mask(tgt_in.size(1)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    for _ in range(50):
        optimizer.zero_grad()
        out = model(dummy_img, tgt_in, mask)
        out = out.reshape(-1, out.size(-1))
        tgt = tgt_out.reshape(-1)
        loss = criterion(out, tgt)
        loss.backward(); optimizer.step()
    assert loss.item() < 0.5, f'Overfit failed: {loss.item():.4f}'
    print('[PASS] Overfit single batch.')

def test_architecture_shapes():
    set_seed(42)
    vocab = Vocabulary()
    vocab.word2idx = {'<<PAD>':0,'<SOS>':1,'<EOS>':2,'<UNK>':3,'a':4,'cat':5}
    vocab.idx2word = {v:k for k,v in vocab.word2idx.items()}; vocab.n_words = len(vocab.word2idx)
    model = ImageCaptioningModel(vocab.n_words, d_model=128, num_layers=2, num_heads=4, dim_ff=256).to(DEVICE)
    img = torch.randn(2,3,224,224).to(DEVICE)
    cap = torch.randint(0,vocab.n_words,(2,10)).to(DEVICE)
    cap_in = cap[:, :-1]
    mask = model.decoder.make_mask(cap_in.size(1)).to(DEVICE)
    out = model(img, cap_in, mask)
    assert out.shape == (2, cap_in.size(1), vocab.n_words), f'Bad shape: {out.shape}'
    print('[PASS] Architecture shapes.')

def test_causal_mask():
    from src.architecture import CaptionDecoder
    dec = CaptionDecoder(vocab_size=10, d_model=32, num_layers=1, num_heads=2, dim_ff=64)
    m = dec.make_mask(5)
    assert m[0,0] == 0 and m[0,1] == float('-inf') and m[4,4] == 0
    print('[PASS] Causal mask.')

def test_train_val_test_split_disjoint():
    from src.preprocessing import split_caps_dict_by_image
    caps = {f'img_{i}.jpg': [f'caption variant {j} for {i}' for j in range(3)] for i in range(60)}
    train_d, val_d, test_d = split_caps_dict_by_image(caps, 0.8, 0.1, 0.1, seed=123)
    kt, kv, kte = set(train_d), set(val_d), set(test_d)
    assert len(kt | kv | kte) == 60
    assert not (kt & kv) and not (kt & kte) and not (kv & kte)
    assert len(train_d) + len(val_d) + len(test_d) == 60
    print('[PASS] Train/val/test split disjoint.')

if __name__ == '__main__':
    test_architecture_shapes(); test_causal_mask(); test_train_val_test_split_disjoint(); test_overfit_one_batch()
    print('\n[OK] All sanity checks passed.')