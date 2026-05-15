import json, random
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from src.config import IMAGE_DIR, FIGURE_DIR
from src.evaluation import Evaluator

class ErrorAnalyzer:
    def __init__(self, model, vocab, eval_loader, references_dict):
        self.evaluator = Evaluator(model, vocab)
        self.eval_loader = eval_loader
        self.references_dict = references_dict
        self.results = None

    def run(self):
        print('[INFO] Running inference for error analysis...')
        self.results = self.evaluator.evaluate(self.eval_loader, self.references_dict)
        return self.results

    def show_random_samples(self, n=6):
        if self.results is None: self.run()
        preds = self.results['predictions']
        samples = random.sample(list(preds.items()), min(n, len(preds)))
        fig, axes = plt.subplots(2, 3, figsize=(15,10)); axes = axes.flatten()
        for ax, (img_name, pred) in zip(axes, samples):
            img = Image.open(IMAGE_DIR / img_name)
            ax.imshow(img)
            refs = self.references_dict.get(img_name, [])[:2]
            ax.set_title(f'PRED: {pred}\nREF: {"\n".join(refs)}', fontsize=9)
            ax.axis('off')
        plt.tight_layout(); save_path = FIGURE_DIR / 'sample_predictions.png'
        plt.savefig(save_path, dpi=150); print(f'[OK] Saved to {save_path}'); plt.show()

    def detect_repeated_words(self):
        if self.results is None: self.run()
        bad = {img: pred for img, pred in self.results['predictions'].items() if len(pred.split()) != len(set(pred.split()))}
        print(f'[INFO] {len(bad)} captions have repeated words.')
        return bad

    def detect_generic_captions(self, generic_words=None):
        generic_words = generic_words or {'a','the','is','are','in','on','of','to'}
        if self.results is None: self.run()
        generic = {img: pred for img, pred in self.results['predictions'].items() if len(set(pred.split()) - generic_words) <= 2}
        print(f'[INFO] {len(generic)} captions are highly generic.')
        return generic

    def generate_report(self):
        if self.results is None: self.run()
        report = {'metrics': {k:v for k,v in self.results.items() if k!='predictions'}, 'repeated': len(self.detect_repeated_words()), 'generic': len(self.detect_generic_captions()), 'total': len(self.results['predictions'])}
        path = FIGURE_DIR / 'error_report.json'
        with open(path, 'w') as f: json.dump(report, f, indent=2)
        print(f'[OK] Report saved to {path}'); return report