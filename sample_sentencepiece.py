"""
Sample with SentencePiece tokenizer
"""
import os
import torch
import sentencepiece as spm
from model import GPTConfig, GPT

out_dir = 'out-eva-bebe-50k'
tokenizer_path = 'data/eva_corpus/eva_bpe_32k.model'
max_new_tokens = 500
temperature = 1.0
top_k = 40
device = 'cuda'
dtype = 'float16'

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
device_type = 'cuda'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load model
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
model.load_state_dict(checkpoint['model'])
model.eval()
model.to(device)

# Load tokenizer
sp = spm.SentencePieceProcessor()
sp.Load(tokenizer_path)

encode = lambda s: sp.EncodeAsIds(s)
decode = lambda l: sp.DecodeIds(l)

# Generate
start_text = "Le"
start_ids = encode(start_text)
if not start_ids:
    start_ids = [1]  # Fallback to SOS token
x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

with torch.no_grad():
    with ctx:
        for _ in range(max_new_tokens):
            if x.size(1) > 512:
                x = x[:, -512:]
            logits, _ = model(x)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, idx_next), dim=1)

output = decode(x[0].tolist())
with open('sample_output.txt', 'w', encoding='utf-8') as f:
    f.write(output)
print("OK: Generated text written to sample_output.txt")
