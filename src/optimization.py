import torch
from src.config import CHECKPOINT_DIR, DEVICE

class ModelOptimizer:
    def __init__(self, model): self.model = model; self.model.eval()

    def quantize_dynamic(self):
        self.model.cpu()
        q = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        torch.save(q.state_dict(), CHECKPOINT_DIR / 'model_quantized.pt')
        print('[OK] Quantized model saved.'); return q

    def export_torchscript_encoder(self, example_image):
        with torch.no_grad():
            traced = torch.jit.trace(self.model.encoder, example_image.unsqueeze(0).to(DEVICE))
        traced.save(str(CHECKPOINT_DIR / 'encoder_traced.pt'))
        print('[OK] TorchScript encoder saved.'); return traced

    def export_onnx(self, example_image):
        try:
            import onnx
            dummy_img = example_image.unsqueeze(0).to(DEVICE)
            dummy_cap = torch.ones(1,1).long().to(DEVICE)
            torch.onnx.export(self.model, (dummy_img, dummy_cap), str(CHECKPOINT_DIR / 'model.onnx'), export_params=True, opset_version=14, do_constant_folding=True, input_names=['image','caption_input'], output_names=['output'], dynamic_axes={'caption_input':{1:'seq_len'}, 'output':{1:'seq_len'}})
            print('[OK] ONNX exported.')
        except ImportError:
            print('[WARN] pip install onnx')
