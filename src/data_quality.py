import re
from pathlib import Path
from collections import Counter
import pandas as pd
from PIL import Image
from src.config import IMAGE_DIR, CAPTION_FILE, PROCESSED_DIR

class DataQualityChecker:
    def __init__(self, image_dir=IMAGE_DIR, caption_file=CAPTION_FILE):
        self.image_dir = Path(image_dir)
        self.caption_file = Path(caption_file)
        self.report = {}

    def check_paths(self):
        assert self.image_dir.exists(), f"[ERR] Image dir not found: {self.image_dir}"
        assert self.caption_file.exists(), f"[ERR] Caption file not found: {self.caption_file}"
        print("[OK] Paths verified.")

    def parse_captions(self):
        rows = []
        with open(self.caption_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split("\t") if "\t" in line else line.split(",", 1)
                if len(parts) == 2:
                    img = parts[0].split("#")[0].strip()
                    rows.append([img, parts[1].strip()])
                else:
                    m = re.match(r"^(\S+)\s+(.*)$", line)
                    if m:
                        img = m.group(1).split("#")[0].strip()
                        rows.append([img, m.group(2).strip()])
        df = pd.DataFrame(rows, columns=["image", "caption"])
        self.report["total_lines"] = len(df)
        self.report["unique_images"] = df["image"].nunique()
        self.report["captions_per_image"] = len(df) / max(df["image"].nunique(), 1)
        print(f"[INFO] Parsed {len(df)} captions for {df['image'].nunique()} images.")
        return df

    def check_image_integrity(self, df):
        corrupted = []
        for name in df["image"].unique():
            path = self.image_dir / name
            if not path.exists():
                corrupted.append((name, "missing")); continue
            try:
                with Image.open(path) as img: img.verify()
            except Exception as e:
                corrupted.append((name, str(e)))
        self.report["corrupted"] = corrupted
        print(f"[{'WARN' if corrupted else 'OK'}] {len(corrupted)} images missing/corrupted.")

    def caption_stats(self, df):
        wc = df["caption"].apply(lambda x: len(str(x).split()))
        df["word_count"] = wc
        self.report["avg_words"] = float(wc.mean())
        self.report["std_words"] = float(wc.std())
        self.report["max_words"] = int(wc.max())
        self.report["min_words"] = int(wc.min())
        print(f"[INFO] Length: {self.report['avg_words']:.1f} ± {self.report['std_words']:.1f}")

    def vocab_diversity(self, df):
        words = " ".join(df["caption"].astype(str)).lower().split()
        self.report["vocab_size"] = len(set(words))
        self.report["top10"] = Counter(words).most_common(10)
        print(f"[INFO] Vocab: {self.report['vocab_size']} unique tokens")

    def save_report(self):
        path = PROCESSED_DIR / "data_quality_report.txt"
        with open(path, "w") as f:
            for k, v in self.report.items(): f.write(f"{k}: {v}\n")
        print(f"[OK] Report saved to {path}")

    def run_all(self):
        self.check_paths(); df = self.parse_captions()
        self.check_image_integrity(df); self.caption_stats(df); self.vocab_diversity(df)
        self.save_report(); return self.report

if __name__ == "__main__":
    DataQualityChecker().run_all()