import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import os
import yaml
import logging
import time

# --- Logging Setup ---
def setup_logging(log_file):
    """Sets up logging to both console and a file."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# --- 1. Configuration Parameters ---
class TransformerConfig:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.d_model = config['d_model'] 
        self.nhead = config['nhead']
        self.num_encoder_layers = config['num_encoder_layers']
        self.num_decoder_layers = config['num_decoder_layers']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']

        self.learning_rate = config['learning_rate']
        self.num_epochs = config['num_epochs']
        self.batch_size = config['batch_size']
        self.eval_interval = config['eval_interval']

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 2. Data Preparation: Unified Vocabulary & Custom Dataset ---
# Special tokens for sequence handling and direction
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
ENG_TO_SANSKRIT_TOKEN = '<eng_to_sanskrit>'
SANSKRIT_TO_ENG_TOKEN = '<sanskrit_to_eng>'
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, ENG_TO_SANSKRIT_TOKEN, SANSKRIT_TO_ENG_TOKEN]

# --- Custom Dataset for Bidirectional Translation ---
class CombinedTextDataset(torch.utils.data.Dataset):
    def __init__(self, eng_filepath, sanskrit_filepath):
        self.eng_lines = self._read_lines(eng_filepath)
        self.sanskrit_lines = self._read_lines(sanskrit_filepath)
        
        assert len(self.eng_lines) == len(self.sanskrit_lines), \
            "English and Sanskrit files must have the same number of lines (aligned translations)."
        
        self.num_pairs = len(self.eng_lines)
        
        all_chars = set()
        for line in self.eng_lines + self.sanskrit_lines:
            all_chars.update(list(line))
        
        self.vocab = SPECIAL_TOKENS + sorted(list(all_chars))
        
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}
        
        self.vocab_size = len(self.vocab)
        
        self.PAD_IDX = self.stoi[PAD_TOKEN]
        self.SOS_IDX = self.stoi[SOS_TOKEN]
        self.EOS_IDX = self.stoi[EOS_TOKEN]
        self.ENG_TO_SANSKRIT_IDX = self.stoi[ENG_TO_SANSKRIT_TOKEN]
        self.SANSKRIT_TO_ENG_IDX = self.stoi[SANSKRIT_TO_ENG_TOKEN]

        max_eng_len = max(len(line) for line in self.eng_lines)
        max_sanskrit_len = max(len(line) for line in self.sanskrit_lines)
        
        self.max_seq_len = max(max_eng_len + 1, max_sanskrit_len + 1) + 2
        
        self.transformer_max_pos_len = self.max_seq_len
    
    def _read_lines(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def __len__(self):
        return self.num_pairs * 2

    def __getitem__(self, idx):
        if idx < self.num_pairs:
            src_lang_idx = idx
            src_text = self.eng_lines[src_lang_idx]
            tgt_text = self.sanskrit_lines[src_lang_idx]
            direction_token_id = self.ENG_TO_SANSKRIT_IDX
        else:
            src_lang_idx = idx - self.num_pairs
            src_text = self.sanskrit_lines[src_lang_idx]
            tgt_text = self.eng_lines[src_lang_idx]
            direction_token_id = self.SANSKRIT_TO_ENG_IDX

        src_tokens = [direction_token_id] + [self.stoi.get(char, self.PAD_IDX) for char in src_text] + [self.EOS_IDX]
        tgt_tokens = [self.SOS_IDX] + [self.stoi.get(char, self.PAD_IDX) for char in tgt_text] + [self.EOS_IDX]

        src_padded = src_tokens + [self.PAD_IDX] * (self.max_seq_len - len(src_tokens))
        tgt_padded = tgt_tokens + [self.PAD_IDX] * (self.max_seq_len - len(tgt_tokens))

        src_padded = src_padded[:self.max_seq_len]
        tgt_padded = tgt_padded[:self.max_seq_len]

        return torch.tensor(src_padded, dtype=torch.long), torch.tensor(tgt_padded, dtype=torch.long)

# --- 3. Masking Functions ---
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
    return mask

def create_mask(src, tgt, PAD_IDX, device):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).bool()

    src_padding_mask = (src == PAD_IDX).to(device)
    tgt_padding_mask = (tgt == PAD_IDX).to(device)
    
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# --- 4. Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# --- 5. Scaled Dot-Product Attention ---
def scaled_dot_product_attention(query, key, value, mask=None, dropout_p=0.0):
    d_k = query.size(-1)
    attn_scores = torch.matmul(query, key.transpose(-2, -1))
    attn_scores = attn_scores / math.sqrt(d_k)
    
    if mask is not None:
        if mask.dtype == torch.bool:
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        else:
            attn_scores = attn_scores + mask
            
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_weights = F.dropout(attn_weights, p=dropout_p)
    
    output = torch.matmul(attn_weights, value)
    
    return output, attn_weights

# --- 6. Multi-Head Attention ---
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_size = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        B, T_q, C = query.size()
        _, T_k, _ = key.size()

        query = self.q_proj(query).view(B, T_q, self.nhead, self.head_size).transpose(1, 2)
        key = self.k_proj(key).view(B, T_k, self.nhead, self.head_size).transpose(1, 2)
        value = self.v_proj(value).view(B, T_k, self.nhead, self.head_size).transpose(1, 2)
        
        combined_mask = attn_mask
        
        if key_padding_mask is not None:
            key_padding_mask_expanded = key_padding_mask.unsqueeze(1).unsqueeze(2)
            if combined_mask is not None:
                combined_mask = combined_mask + key_padding_mask_expanded.float().masked_fill(key_padding_mask_expanded, float('-inf'))
            else:
                combined_mask = key_padding_mask_expanded.float().masked_fill(key_padding_mask_expanded, float('-inf'))

        attn_output, attn_weights_raw = scaled_dot_product_attention(
            query, key, value, combined_mask, self.dropout.p
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_q, C)
        output = self.out_proj(attn_output)
        
        return output, attn_weights_raw.mean(dim=1)

# --- 7. Position-wise Feed-Forward Networks (FFN) ---
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear2(F.relu(self.linear1(x))))

# --- 8. Encoder Layer ---
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, dim_feedforward, dropout=dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        
        ff_output = self.feed_forward(src)
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        
        return src

# --- 9. Decoder Layer ---
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, dim_feedforward, dropout=dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, 
                tgt_mask: torch.Tensor = None, memory_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None, memory_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        attn1_output, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(attn1_output)
        tgt = self.norm1(tgt)
        
        attn2_output, _ = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(attn2_output)
        tgt = self.norm2(tgt)
        
        ff_output = self.feed_forward(tgt)
        tgt = tgt + self.dropout3(ff_output)
        tgt = self.norm3(tgt)
        
        return tgt

# --- 10. Full Transformer Model ---
class Transformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_encoder_layers: int,
                 num_decoder_layers: int, dim_feedforward: int, dropout: float, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        
        self.tok_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_decoder_layers)
        ])
        
        self.output_head = nn.Linear(d_model, vocab_size)

        self.output_head.weight = self.tok_embedding.weight
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, 
                src_mask: torch.Tensor, tgt_mask: torch.Tensor,
                src_key_padding_mask: torch.Tensor, tgt_key_padding_mask: torch.Tensor) -> torch.Tensor:
        
        src_emb = self.pos_encoder(self.tok_embedding(src))
        tgt_emb = self.pos_encoder(self.tok_embedding(tgt))

        encoder_output = src_emb
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        decoder_output = tgt_emb
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(tgt=decoder_output, memory=encoder_output, 
                                            tgt_mask=tgt_mask, 
                                            memory_mask=None,
                                            tgt_key_padding_mask=tgt_key_padding_mask, 
                                            memory_key_padding_mask=src_key_padding_mask) 

        logits = self.output_head(decoder_output)
        
        return logits
    
# --- 11. Training Loop ---
def train_model(model, dataloader, optimizer, criterion, config, dataset, logger):
    logger.info("\n--- Starting Training ---")
    
    epoch_metrics = []

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (src, tgt) in enumerate(dataloader):
            src = src.to(config.device)
            tgt = tgt.to(config.device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, dataset.PAD_IDX, config.device)
            
            logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)
            
            loss = criterion(logits.reshape(-1, dataset.vocab_size), tgt_output.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        
        epoch_metrics.append({'epoch': epoch + 1, 'loss': avg_loss})

        if (epoch + 1) % config.eval_interval == 0 or epoch == config.num_epochs - 1:
            logger.info(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {avg_loss:.4f}")
    
    logger.info("\n--- Training Complete ---")
    logger.info(f"\n--- Epoch Metrics Log ---")
    logger.info(str(epoch_metrics))


# --- 12. Inference (Bidirectional Translation) Function ---
@torch.no_grad()
def translate_sentence_bidirectional(model, sentence: str, direction: str, max_output_len: int = 20, dataset=None, config=None):
    model.eval()
    stoi_map = dataset.stoi
    itos_map = dataset.itos
    
    if direction == "eng_to_sanskrit":
        direction_token_id = dataset.ENG_TO_SANSKRIT_IDX
    elif direction == "sanskrit_to_eng":
        direction_token_id = dataset.SANSKRIT_TO_ENG_IDX
    else:
        raise ValueError("Direction must be 'eng_to_sanskrit' or 'sanskrit_to_eng'")

    src_tokens_list = [direction_token_id] + [stoi_map.get(char, dataset.PAD_IDX) for char in sentence] + [dataset.EOS_IDX]
    src_tokens_list_padded = src_tokens_list + [dataset.PAD_IDX] * (dataset.max_seq_len - len(src_tokens_list))
    src_tokens = torch.tensor(src_tokens_list_padded[:dataset.max_seq_len], dtype=torch.long).unsqueeze(0).to(config.device)

    src_mask, _, src_padding_mask, _ = create_mask(src_tokens, src_tokens, dataset.PAD_IDX, config.device)
    
    src_emb = model.pos_encoder(model.tok_embedding(src_tokens))
    encoder_output = src_emb
    for encoder_layer in model.encoder_layers:
        encoder_output = encoder_layer(encoder_output, src_mask=src_mask, src_key_padding_mask=src_padding_mask)

    decoder_input = torch.full((1, 1), dataset.SOS_IDX, dtype=torch.long, device=config.device)
    generated_tokens = []

    for i in range(max_output_len):
        tgt_mask = generate_square_subsequent_mask(decoder_input.size(1)).to(config.device)
        tgt_key_padding_mask = (decoder_input == dataset.PAD_IDX).to(config.device)

        logits = model(src_tokens, decoder_input, 
                        src_mask, tgt_mask, 
                        src_padding_mask, tgt_key_padding_mask)
        
        next_token_logits = logits[:, -1, :]
        predicted_token_id = next_token_logits.argmax(dim=-1)
        
        generated_tokens.append(predicted_token_id.item())
        
        if predicted_token_id.item() == dataset.EOS_IDX:
            break
        
        decoder_input = torch.cat([decoder_input, predicted_token_id.unsqueeze(0)], dim=1)
        
    def decode_sentence(tokens, itos_map):
        # Corrected: Handle tokens that might be out of vocabulary
        decoded_chars = []
        for t in tokens:
            t_int = t.item()
            if t_int not in [dataset.SOS_IDX, dataset.EOS_IDX, dataset.PAD_IDX] and t_int in itos_map:
                decoded_chars.append(itos_map[t_int])
        return "".join(decoded_chars)
        
    return decode_sentence(torch.tensor(generated_tokens), itos_map=itos_map)

# --- Main Execution ---
if __name__ == "__main__":
    # --- Create Dummy Files ---
    dummy_eng_data = [
        "hello", "world", "cat", "dog", "apple",
        "water", "sun", "moon", "good", "bad",
        "I am a student.", "He is a teacher.", "She loves to read.",
        "The bird sings.", "We learn Python."
    ]
    dummy_sanskrit_data = [
        "नमस्ते", "विश्व", "बिल्ली", "कुत्ता", "सेब",
        "पानी", "सूर्य", "चंद्रमा", "अच्छा", "बुरा",
        "अहं छात्रोऽस्मि।", "सः शिक्षकः अस्ति।", "सा पठितुं रोचते।",
        "पक्षी गायति।", "वयं पाइथनं शिक्षामहे。"
    ]
    with open('english.txt', 'w', encoding='utf-8') as f:
        for line in dummy_eng_data: f.write(line + '\n')
    with open('sanskrit.txt', 'w', encoding='utf-8') as f:
        for line in dummy_sanskrit_data: f.write(line + '\n')
    
    # --- Initialize ---
    config = TransformerConfig('configs/config.yaml')
    
    # --- Logging Setup ---
    if not os.path.exists('logs'):
        os.makedirs('logs')
    log_file_name = f"transformer_training_log_{int(time.time())}.log"
    logger = setup_logging(os.path.join('logs', log_file_name))
    logger.info(f"Using device: {config.device}")


    dataset = CombinedTextDataset('english.txt', 'sanskrit.txt')
    VOCAB_SIZE_COMBINED = dataset.vocab_size
    PAD_IDX = dataset.PAD_IDX
    SOS_IDX = dataset.SOS_IDX
    EOS_IDX = dataset.EOS_IDX
    MAX_SEQ_LEN_TRANSFORMER = dataset.transformer_max_pos_len

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    logger.info(f"Combined Vocab Size: {dataset.vocab_size}")
    logger.info(f"Max sentence length in dataset: {dataset.max_seq_len}")
    logger.info(f"Transformer Positional Encoding Max Length: {dataset.transformer_max_pos_len}")


    model = Transformer(
        vocab_size=VOCAB_SIZE_COMBINED,
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_seq_len=MAX_SEQ_LEN_TRANSFORMER
    ).to(config.device)

    logger.info(f"\nModel initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f} M parameters on {config.device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    train_model(model, dataloader, optimizer, criterion, config, dataset, logger)

    logger.info("\n--- Testing Bidirectional Translation ---")

    logger.info("\nEnglish to Sanskrit:")
    test_eng_sentences = ["hello", "cat", "I am a student."]
    for sentence in test_eng_sentences:
        translated = translate_sentence_bidirectional(model, sentence, "eng_to_sanskrit", dataset=dataset, config=config)
        logger.info(f"'{sentence}' -> '{translated}'")

    logger.info("\nSanskrit to English:")
    test_sanskrit_sentences = ["नमस्ते", "बिल्ली", "अहं छात्रोऽस्मि।"]
    for sentence in test_sanskrit_sentences:
        translated = translate_sentence_bidirectional(model, sentence, "sanskrit_to_eng", dataset=dataset, config=config)
        logger.info(f"'{sentence}' -> '{translated}'")

    os.remove('english.txt')
    os.remove('sanskrit.txt')
    logger.info("\nDummy files cleaned up.")
    logger.info("\n--- Training Complete ---")
