import sys, io, base64
from pathlib import Path
from flask import Flask, request, render_template, jsonify
from PIL import Image
import torch, torchvision.transforms as T
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import DEVICE, CHECKPOINT_DIR
from src.architecture import ImageCaptioningModel
from src.evaluation import Evaluator

app = Flask(__name__, template_folder='templates', static_folder='static')

print('[INFO] Loading checkpoint...')
checkpoint = torch.load(CHECKPOINT_DIR / 'best_model.pt', map_location=DEVICE, weights_only=False)
vocab = checkpoint['vocab']
model = ImageCaptioningModel(vocab.n_words, 512, 6, 8, 2048, 0.5, 14).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict']); model.eval()
evaluator = Evaluator(model, vocab)

_transform = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

@app.route('/')
def home(): return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files: return jsonify({'error':'No image uploaded'}), 400
    img = Image.open(request.files['image'].stream).convert('RGB')
    tensor = _transform(img)
    caption = evaluator.beam_search_decode(tensor)
    buf = io.BytesIO(); img.save(buf, format='JPEG'); b64 = base64.b64encode(buf.getvalue()).decode()
    return jsonify({'caption': caption, 'image': f'data:image/jpeg;base64,{b64}'})

@app.route('/api/caption', methods=['POST'])
def api_caption():
    if 'image' not in request.files: return jsonify({'error':'No image uploaded'}), 400
    img = Image.open(request.files['image'].stream).convert('RGB')
    tensor = _transform(img)
    return jsonify({'caption': evaluator.beam_search_decode(tensor)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
