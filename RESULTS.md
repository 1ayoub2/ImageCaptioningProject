## Latest Training Run

| Metric  | Score  |
|---------|--------|
| BLEU-1  | 0.5942 |
| BLEU-4  | 0.1847 |
| METEOR  | 0.4180 |
| CIDEr   | 0.3015 |

**Architecture**: ResNet101 + Transformer (6L · 8H · 512d)  
**Dataset**: Flickr8k | **Epochs**: 23 (early stop)  
**Config**: LR=1e-4 · Warmup=1000 · CNN unfrozen ep5 · Beam=3  
