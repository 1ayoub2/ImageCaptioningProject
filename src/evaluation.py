import torch, numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import nltk
from src.config import DEVICE, BEAM_WIDTH, MAX_GEN_LEN
from src.preprocessing import clean_caption
try: nltk.data.find('wordnet')
except: nltk.download('wordnet'); nltk.download('omw-1.4')

class Evaluator:
    def __init__(self, model, vocab, beam_width=BEAM_WIDTH):
        self.model = model.to(DEVICE); self.model.eval()
        self.vocab = vocab; self.beam_width = beam_width
        self.idx2word = vocab.idx2word
        self.end_token = vocab.word2idx['<EOS>']
        self.start_token = vocab.word2idx['<SOS>']

    def greedy_decode(self, image, max_len=MAX_GEN_LEN):
        with torch.no_grad():
            memory = self.model.encoder(image.unsqueeze(0).to(DEVICE))
            ys = torch.ones(1,1).fill_(self.start_token).long().to(DEVICE)
            for _ in range(max_len-1):
                tgt_mask = self.model.decoder.make_mask(ys.size(1)).to(DEVICE)
                out = self.model.decoder(ys, memory, tgt_mask)
                _, next_word = torch.max(out[:,-1], dim=1)
                ys = torch.cat([ys, next_word.view(1, 1)], dim=1)
                if next_word.item() == self.end_token: break
        return self._tensor_to_sentence(ys[0])

    def beam_search_decode(self, image, max_len=MAX_GEN_LEN, beam_width=None):
        beam_width = beam_width or self.beam_width
        with torch.no_grad():
            memory = self.model.encoder(image.unsqueeze(0).to(DEVICE))
            sequences = [(0.0, torch.ones(1,1).fill_(self.start_token).long().to(DEVICE))]
            for _ in range(max_len-1):
                all_candidates = []
                for score, seq in sequences:
                    if seq[0,-1].item() == self.end_token:
                        all_candidates.append((score, seq)); continue
                    tgt_mask = self.model.decoder.make_mask(seq.size(1)).to(DEVICE)
                    out = self.model.decoder(seq, memory, tgt_mask)
                    log_probs = torch.log_softmax(out[:,-1], dim=-1)
                    topk_log, topk_words = log_probs.topk(beam_width)
                    for log_prob, word in zip(topk_log[0], topk_words[0]):
                        penalty = ((5 + seq.size(1)) / 6.0) ** 0.6
                        new_score = (score + log_prob.item()) / penalty
                        new_seq = torch.cat([seq, word.unsqueeze(0).unsqueeze(0)], dim=1)
                        all_candidates.append((new_score, new_seq))
                ordered = sorted(all_candidates, key=lambda x: x[0], reverse=True)
                sequences = ordered[:beam_width]
                if all(seq[0,-1].item() == self.end_token for _, seq in sequences): break
            return self._tensor_to_sentence(sequences[0][1][0])

    def _tensor_to_sentence(self, tensor):
        words = []
        for idx in tensor.tolist():
            word = self.idx2word.get(idx, '<UNK>')
            if word == '<EOS>': break
            if word not in ('<SOS>', '<<PAD>', '<UNK>'): words.append(word)
        return ' '.join(words)

    def evaluate(
        self,
        data_loader,
        references_dict,
        *,
        desc="Evaluating",
        verbose=True,
        return_predictions=True,
    ):
        predictions = {} if return_predictions else None
        references = []
        hypotheses = []
        for images, _, img_names in tqdm(data_loader, desc=desc):
            for i, img_name in enumerate(img_names):
                pred = self.beam_search_decode(images[i])
                if predictions is not None:
                    predictions[img_name] = pred
                refs = [clean_caption(r).split() for r in references_dict.get(img_name, [])]
                if refs:
                    references.append(refs)
                    hypotheses.append(pred.split())
        smooth = SmoothingFunction().method1
        bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smooth)
        bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
        meteor_scores = [
            max(meteor_score([" ".join(r) for r in refs], " ".join(hyp)) for r in refs)
            for refs, hyp in zip(references, hypotheses)
        ]
        meteor = np.mean(meteor_scores)
        if verbose:
            print(f"\n[RESULTS] BLEU-1: {bleu1:.4f} | BLEU-4: {bleu4:.4f} | METEOR: {meteor:.4f}")
        out = {"bleu1": bleu1, "bleu4": bleu4, "meteor": meteor}
        if predictions is not None:
            out["predictions"] = predictions
        return out