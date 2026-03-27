"""
Sample from a trained model with optional TurboQuant KV cache compression.

Standard sampling:
$ python sample.py --out_dir=out

With TurboQuant (reduces KV cache memory ~5x):
$ python sample.py --out_dir=out --use_turboquant=True --turboquant_bits=3

Compare quality with/without TurboQuant:
$ python sample.py --out_dir=out --compare_turboquant=True
"""
import os
import time
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume'  # 'resume' or 'gpt2' variant
out_dir = 'out'
start = "\n"
num_samples = 10
max_new_tokens = 500
temperature = 0.8
top_k = 200
seed = 1337
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
# TurboQuant settings
use_turboquant = False
turboquant_bits = 3
compare_turboquant = False  # if True, generate with both modes and compare
exec(open('configurator.py').read())
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint['model_args']
    # Inject TurboQuant settings
    model_args['use_turboquant'] = use_turboquant
    model_args['turboquant_bits'] = turboquant_bits
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0, use_turboquant=use_turboquant, turboquant_bits=turboquant_bits))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])


def generate_and_time(use_tq, label=""):
    """Generate samples and measure time."""
    torch.manual_seed(seed)
    total_tokens = 0
    t0 = time.time()
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, use_turboquant=use_tq)
                text = decode(y[0].tolist())
                total_tokens += max_new_tokens
                if label:
                    print(f"[{label}] Sample {k + 1}:")
                print(text)
                print('---------------')
    elapsed = time.time() - t0
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
    print(f"\n{'=' * 40}")
    print(f"{label}: {num_samples} samples, {total_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")
    if use_tq:
        from turboquant import TurboQuantKVCompressor
        head_dim = model.config.n_embd // model.config.n_head
        comp = TurboQuantKVCompressor(head_dim, turboquant_bits)
        info = comp.memory_savings_info(max_new_tokens, model.config.n_head)
        print(f"TurboQuant {turboquant_bits}-bit: {info['compression_ratio']:.1f}x KV cache compression")
    print(f"{'=' * 40}\n")
    return elapsed


if compare_turboquant:
    print("=" * 60)
    print("COMPARISON: Standard vs TurboQuant inference")
    print("=" * 60)
    print("\n--- Standard (FP16/BF16) ---\n")
    t_standard = generate_and_time(use_tq=False, label="Standard")
    print("\n--- TurboQuant %d-bit ---\n" % turboquant_bits)
    t_tq = generate_and_time(use_tq=True, label="TurboQuant")
    print("=" * 60)
    print(f"Standard:   {t_standard:.2f}s")
    print(f"TurboQuant: {t_tq:.2f}s")
    if t_standard > 0:
        print(f"Speedup:    {t_standard / t_tq:.2f}x" if t_tq > 0 else "N/A")
    print("=" * 60)
else:
    generate_and_time(use_tq=use_turboquant, label="TurboQuant" if use_turboquant else "Standard")
