import math, torch, torch.nn as nn, torchvision

class ImageEncoder(nn.Module):
    def __init__(self, encoded_image_size=14, d_model=512, dropout=0.5):
        super().__init__()
        self.enc_size = encoded_image_size
        weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V1
        resnet = torchvision.models.resnet101(weights=weights)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fc = nn.Linear(resnet.fc.in_features, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, images):
        x = self.resnet(images)
        x = self.pool(x)
        x = x.permute(0,2,3,1).view(x.size(0), -1, x.size(1))
        x = self.fc(x)
        return self.norm(self.dropout(x))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:,0::2] = torch.sin(pos * div)
        pe[:,1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, dim_ff=2048, dropout=0.5, max_len=50):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        layer = nn.TransformerDecoderLayer(d_model, num_heads, dim_ff, dropout, batch_first=True, norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.fc_out.weight = self.embedding.weight
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def make_mask(self, sz):
        return torch.triu(torch.ones(sz, sz), diagonal=1).masked_fill(torch.triu(torch.ones(sz, sz), diagonal=1) == 1, float("-inf"))

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        tgt_emb = self.dropout(self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model)))
        out = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return self.fc_out(out)

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, dim_ff=2048, dropout=0.5, encoded_size=14):
        super().__init__()
        self.encoder = ImageEncoder(encoded_size, d_model, dropout)
        self.decoder = CaptionDecoder(vocab_size, d_model, num_layers, num_heads, dim_ff, dropout)

    def forward(self, images, captions, caption_mask=None, caption_padding_mask=None):
        memory = self.encoder(images)
        return self.decoder(captions, memory, caption_mask, caption_padding_mask)