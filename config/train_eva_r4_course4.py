# Course 4 — EVA avec Corpus_R4 (313M tokens, reprise depuis 50k)
# Tokenizer : eva_bpe_32k (vocab_size=32000, compatible)
# Données : data/corpus_r4/train.bin + val.bin

out_dir = 'out-eva-r4-course4'
eval_interval = 500
eval_iters = 50
log_interval = 50
always_save_checkpoint = True

# Reprendre depuis le modèle 50k
init_from = 'resume'

dataset = 'corpus_r4'
gradient_accumulation_steps = 4
batch_size = 8
block_size = 512

# Architecture identique au modèle 50k (41M params)
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1
bias = False

# LR schedule sur 50k nouvelles itérations (reprise iter 50k → cible 100k)
learning_rate = 3e-4        # réduit vs 1e-3 (modèle partiellement convergé)
max_iters = 100000
lr_decay_iters = 98000
min_lr = 3e-5
beta2 = 0.99
warmup_iters = 200
weight_decay = 1e-1
grad_clip = 1.0

device = 'cuda'
dtype = 'float16'
compile = False
