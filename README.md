# Image Captioning — CNN + Transformer

End-to-end pipeline for Flickr8k using **ResNet101 spatial encoder** + **Transformer decoder**.

## Pipeline
1. **Data Quality** — integrity checks, caption stats
2. **Preprocessing** — vocabulary, Dataset, DataLoader
3. **Balancing** — label smoothing + inverse frequency weights
4. **Augmentation** — safe transforms (no rotation/zoom)
5. **Architecture** — ResNet101 (14×14 grid) → TransformerDecoder with causal mask
6. **Training** — AdamW + warmup + cosine decay + grad clip + CNN freeze/unfreeze
7. **Evaluation** — BLEU-1/4, METEOR, beam search
8. **Error Analysis** — visualize, detect mode collapse
9. **Optimization** — quantization, TorchScript, ONNX
10. **Deployment** — Flask web UI + REST API

## Quick Start (Windows)
Open a PowerShell terminal in the project folder and run:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
# Place Flickr8k in data\raw\
python -m src.data_quality
python tests\test_pipeline.py
python main.py
python api\app.py
```
