import torch, torch.nn as nn, torch.nn.functional as F
from collections import Counter

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, padding_idx, smoothing=0.1):
        super().__init__()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        log_probs = F.log_softmax(x, dim=-1)
        nll = F.nll_loss(log_probs, target, reduction="none", ignore_index=self.padding_idx)
        smooth = -log_probs.mean(dim=-1)
        loss = self.confidence * nll + self.smoothing * smooth
        mask = target != self.padding_idx
        return loss[mask].mean() if mask.sum() > 0 else loss.sum() * 0.0

class InverseFrequencyWeights:
    def __init__(self, captions, vocab, pad_idx):
        self.vocab = vocab; self.pad_idx = pad_idx
        self.weights = self._compute(captions)

    def _compute(self, captions):
        freq = Counter()
        for cap in captions:
            for w in cap.split():
                idx = self.vocab.word2idx.get(w, self.vocab.word2idx['<UNK>'])
                if idx != self.pad_idx: freq[idx] += 1
        weights = torch.ones(self.vocab.n_words)
        total = sum(freq.values())
        for idx, cnt in freq.items(): weights[idx] = 1.0 / (cnt / total)
        weights[self.pad_idx] = 0.0
        return weights / weights.mean()
