# 🧠 Mini Generative Pre-Trained Transformer (Mini-GPT)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Research-orange?style=for-the-badge)

**A from-scratch implementation of a GPT-style decoder-only Transformer trained on literary and web text.**

</div>

---

## 📖 Table of Contents

1. [Project Overview](#-project-overview)
2. [Architecture Deep Dive](#-architecture-deep-dive)
   - [Tokenization & Vocabulary](#tokenization--vocabulary)
   - [Token & Positional Embeddings](#token--positional-embeddings)
   - [Self-Attention Head](#self-attention-head)
   - [Multi-Head Attention](#multi-head-attention)
   - [Feed-Forward Network](#feed-forward-network)
   - [Transformer Block](#transformer-block)
   - [Full GPT Language Model](#full-gpt-language-model)
   - [Text Generation (Autoregressive Decoding)](#text-generation-autoregressive-decoding)
3. [Data Pipeline](#-data-pipeline)
4. [Training Pipeline](#-training-pipeline)
   - [Hyperparameters](#hyperparameters)
   - [Memory-Mapped I/O](#memory-mapped-io)
   - [Mixed-Precision Training (AMP)](#mixed-precision-training-amp)
   - [Gradient Clipping](#gradient-clipping)
   - [torch.compile()](#torchcompile)
   - [Loss Estimation](#loss-estimation)
5. [Project Structure](#-project-structure)
6. [Setup & Installation](#-setup--installation)
7. [Usage](#-usage)
   - [Step 1 — Data Extraction](#step-1--data-extraction)
   - [Step 2 — Training](#step-2--training)
   - [Step 3 — Inference](#step-3--inference)
8. [Research Notebook](#-research-notebook)
9. [Saved Artifacts](#-saved-artifacts)
10. [GPU / Device Support](#-gpu--device-support)
11. [Future Work & Improvements](#-future-work--improvements)
12. [References](#-references)

---

## 🔭 Project Overview

This project is a **ground-up implementation** of a GPT (Generative Pre-Trained Transformer) built entirely with **raw PyTorch** — no HuggingFace model weights, no pre-built Transformer layers. The goal is a thorough, transparent understanding of how large language models actually work at the tensor level.

The model follows the **decoder-only Transformer** paradigm introduced in the original GPT paper (Radford et al., 2018) and popularised in GPT-2/GPT-3. It is trained on:

- 📚 **The Wizard of Oz** — a classic literary text (`wizard_of_oz.txt`) used for initial character-level prototyping.
- 🌊 **OpenWebText** — a large web-crawled corpus of `.xz`-compressed files, sub-sampled and extracted via `extract.py`.

The entire pipeline — from raw text → character tokenization → batching → Transformer forward pass → loss computation → checkpoint saving — is hand-written and fully inspectable.

---

## 🏛 Architecture Deep Dive

The model is a **decoder-only, causal language model** with a classic stacked-Transformer design. Below is an annotated walkthrough of every component.

```
Input tokens (B, T)
       │
       ▼
Token Embedding (B, T, C)   +   Positional Embedding (T, C)
       │
       ▼
┌─────────────────────────┐
│   Transformer Block × 6  │
│  ┌─────────────────────┐│
│  │ LayerNorm            ││
│  │ Multi-Head Attention ││   ← 8 heads, causal mask
│  │ Residual Add         ││
│  │ LayerNorm            ││
│  │ Feed-Forward (×4)    ││   ← expand → ReLU → project
│  │ Residual Add         ││
│  └─────────────────────┘│
└─────────────────────────┘
       │
       ▼
Final LayerNorm
       │
       ▼
LM Head Linear (C → vocab_size)
       │
       ▼
Logits → Cross-Entropy Loss (training) / Softmax + Sample (inference)
```

---

### Tokenization & Vocabulary

The model uses **character-level tokenization** — each unique character in the corpus maps to an integer index.

```python
chars = sorted(list(set(text)))
string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join(int_to_string[i] for i in l)
```

For the large OpenWebText corpus, the full character vocabulary is assembled in parallel by `extract.py` and serialized to `vocab.txt`. This avoids scanning the entire corpus again at training time.

> **Why character-level?** It keeps `vocab_size` tiny (typically a few hundred), makes the tokenizer dependency-free, and makes every design decision visible. The tradeoff is longer sequence lengths needed to represent the same semantic content.

---

### Token & Positional Embeddings

```python
self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)   # maps token id → dense vector
self.position_embedding_table = nn.Embedding(block_size, n_embd)   # maps position → dense vector
```

Both embeddings are **learned** (not fixed sinusoidal, unlike the original Attention Is All You Need paper). During the forward pass they are summed:

```python
tok_emb = self.token_embedding_table(index)                          # (B, T, C)
pos_emb = self.position_embedding_table(torch.arange(T, device=…))  # (T, C)
x = tok_emb + pos_emb                                               # (B, T, C)
```

This gives every token both a semantic identity and an awareness of *where* it sits in the sequence — crucial because self-attention is inherently permutation-invariant without positional signals.

---

### Self-Attention Head

Each attention head learns its own linear projections for **Keys**, **Queries**, and **Values**:

```python
self.key   = nn.Linear(n_embd, head_size, bias=False)
self.query = nn.Linear(n_embd, head_size, bias=False)
self.value = nn.Linear(n_embd, head_size, bias=False)
```

The **scaled dot-product attention** computation:

```
Attention(Q, K, V) = softmax( QKᵀ / √dₖ ) · V
```

is implemented as:

```python
k = self.key(x)                                         # (B, T, hs)
q = self.query(x)                                       # (B, T, hs)
wei = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)  # (B, T, T)
```

**Causal masking** is applied via a lower-triangular mask registered as a non-parameter buffer:

```python
self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)
```

Positions in the future are masked to `-inf` before softmax, which zeroes them out — ensuring each token can only attend to itself and tokens that preceded it (autoregressive property).

Finally, a dropout is applied to attention weights before computing the weighted sum:

```python
wei = self.dropout(wei)
v   = self.value(x)
out = wei @ v           # (B, T, hs)
```

---

### Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
```

`n_head = 8` heads run **independently and in parallel**, each attending to different sub-spaces of the embedding. Their outputs are concatenated along the channel dimension (restoring width to `n_embd`) and then projected with one more linear layer.

| Param | Value |
|---|---|
| `n_head` | 8 |
| `head_size` | `n_embd // n_head` = 32 |
| Concat output dim | 8 × 32 = 256 |
| After projection | 256 (= `n_embd`) |

---

### Feed-Forward Network

A standard 2-layer MLP with a 4× expansion is applied position-wise after attention:

```python
self.net = nn.Sequential(
    nn.Linear(n_embd, 4 * n_embd),   # expand:    256 → 1024
    nn.ReLU(),
    nn.Linear(4 * n_embd, n_embd),   # contract: 1024 → 256
    nn.Dropout(dropout),
)
```

This mirrors the FFN in the original Transformer paper. The expansion introduces non-linearity and allows the model to compute more complex per-token transformations after the global mixing done by attention.

---

### Transformer Block

Each block applies the **Pre-LN** (Pre-Layer-Normalization) variant, which has been found to be more stable than the original Post-LN:

```python
def forward(self, x):
    x = x + self.sa(self.ln1(x))    # Self-Attention sub-layer with residual
    x = x + self.ffwd(self.ln2(x))  # FFN sub-layer with residual
    return x
```

**Residual connections** allow gradients to flow through directly during backprop, mitigating the vanishing gradient problem in deep networks. `n_layer = 6` such blocks are stacked.

---

### Full GPT Language Model

```python
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        ...
        self.blocks  = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)
```

**Weight initialization** follows the GPT-2 convention — Normal distribution with `std=0.02`, zeros for all biases:

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

**Loss computation** uses cross-entropy on flattened logits:

```python
B, T, C = logits.shape
loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
```

---

### Text Generation (Autoregressive Decoding)

```python
def generate(self, index, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = index[:, -block_size:]          # trim to context window
        logits, _ = self.forward(idx_cond)
        logits_last = logits[:, -1, :]             # only the last-position logits
        probs = F.softmax(logits_last, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)  # sample
        index = torch.cat((index, idx_next), dim=1)
    return index
```

Text generation is fully **autoregressive** — at each step the model predicts the probability distribution over the vocabulary for the *next* character, samples from it stochastically (`torch.multinomial`), appends the result to the context, and loops. If the context would exceed `block_size = 128`, it is left-truncated to always stay within the positional embedding table's range.

---

## 🗂 Data Pipeline

### Source 1 — Wizard of Oz (`wizard_of_oz.txt`)

A ~237 KB literary text used for the prototyping phase and initial character-vocabulary construction. The model can begin training on this corpus alone for quick experimentation.

### Source 2 — OpenWebText (via `extract.py`)

A large-scale dataset of internet text distributed as many `.xz`-compressed files. The `extract.py` script handles the complete ETL:

| Stage | Description |
|---|---|
| **Discovery** | Scans a folder for all `.xz` files using `xz_file_in_dir()` |
| **Splitting** | 90 / 10 train-validation split by file count |
| **Sampling** | Only 1% of files are sampled per run (`sampling_rate = 0.01`), keeping it manageable |
| **Parallel Decompression** | Uses `ProcessPoolExecutor` with `max_workers=4` to decompress files concurrently |
| **Safe Merging** | Each worker writes to an isolated temp file; the main process sequentially merges into `output_train.txt` / `output_val.txt` |
| **Vocab Building** | Accumulates character sets across all files; final sorted vocab saved to `vocab.txt` |

```
OpenWebText .xz files
         │
   ProcessPoolExecutor (4 workers)
         │ lzma.open() + text decode
         │ write to unique temp .part files
         ▼
   Main process merges temp files
         │
         ▼
  output_train.txt (~385 MB)
  output_val.txt   (~41 MB)
  vocab.txt
```

### Memory-Mapped Batch Sampling (`get_random_chunk`)

At training time, reading the full 385 MB training file into RAM every step would be prohibitively slow. Instead, `mmap` (memory-mapped file I/O) is used:

```python
def get_random_chunk(split):
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            start_pos = random.randint(0, file_size - block_size * batch_size)
            mm.seek(start_pos)
            block = mm.read(block_size * batch_size - 1)
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            data = torch.tensor(encode(decoded_block), dtype=torch.long)
    return data
```

`mmap` maps the file directly into the virtual address space — the OS pages in only the relevant disk sectors on demand, making random seeks across gigabyte-scale files essentially instant. This is what makes it practical to train on the full OpenWebText corpus without loading it into RAM.

---

## ⚙️ Training Pipeline

### Hyperparameters

| Hyperparameter | Value | Description |
|---|---|---|
| `batch_size` | 32 | Number of independent sequences per training step |
| `block_size` | 128 | Context/sequence length (tokens) seen at once |
| `max_iters` | 20,000 | Total number of gradient update steps |
| `learning_rate` | 3e-4 | AdamW learning rate |
| `eval_iters` | 200 | Number of batches averaged when estimating loss |
| `n_embd` | 256 | Embedding / model dimension |
| `n_head` | 8 | Number of attention heads |
| `n_layer` | 6 | Number of stacked Transformer blocks |
| `dropout` | 0.1 | Dropout probability throughout the model |

**Approximate parameter count:**
- Token embedding: `vocab_size × 256`
- Position embedding: `128 × 256 = 32,768`
- Per block ≈ `4 × 256² + 4 × 256 × 1024` ≈ ~1.3M parameters
- 6 blocks + embeddings + LM head ≈ **~10–15M parameters** (varies with vocab size)

---

### Memory-Mapped I/O

See [above](#memory-mapped-batch-sampling-get_random_chunk). The `get_batch` function calls `get_random_chunk`, then randomly slices `batch_size` examples from the returned chunk and stacks them into `(B, T)` tensors for both `x` (input) and `y` (target, shifted by 1).

---

### Mixed-Precision Training (AMP)

To accelerate training on compatible NVIDIA GPUs (Ampere+, e.g., RTX 30xx/40xx), **Automatic Mixed Precision (AMP)** is used:

```python
scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
    logits, loss = model(xb, yb)

optimizer.zero_grad(set_to_none=True)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

- `torch.autocast` casts eligible operations to `float16`, halving memory bandwidth and exploiting Tensor Cores.
- `GradScaler` scales the loss upward before backward pass to prevent underflow in `float16` gradients, then unscales before the optimizer step.
- `torch.set_float32_matmul_precision("high")` additionally enables TF32 on Ampere GPUs for FP32 matmul operations.
- `optimizer.zero_grad(set_to_none=True)` is used instead of `.zero_grad()` to free gradient tensors from memory entirely between steps.

---

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

Clips the global L2 norm of all gradients to 1.0. This is standard practice for Transformer training to prevent rare but catastrophic gradient explosions from destabilizing the model.

---

### torch.compile()

```python
try:
    model = torch.compile(model)
    print("✅ Using torch.compile() optimized mode.")
except Exception as e:
    print(f"⚠️ torch.compile() failed ({e}); running in eager mode.")
```

`torch.compile()` (introduced in PyTorch 2.0) traces the model graph and applies back-end optimizations (kernel fusion, memory planning, etc.) using Triton. On compatible hardware this can yield **1.3–2×** speedups on the training loop. The try/except pattern ensures the code falls back gracefully to eager mode on Windows (where Triton compilation can be limited) or on CPU.

---

### Loss Estimation

```python
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss
        out[split] = losses.mean().detach().cpu()
    model.train()
    return out
```

Every **100 steps**, the model is switched to `eval()` mode (disabling dropout) and loss is averaged over 200 random batches from each split. This produces a statistically stable estimate of generalization performance. `@torch.no_grad()` disables gradient tracking during evaluation, halving memory usage for that phase.

---

## 📁 Project Structure

```
Mini_Generative_Pretrained_Transformer/
│
├── training.py            # Full model definition + training loop
├── main.py                # Inference / text generation script
├── extract.py             # Parallel OpenWebText extraction & vocab building
│
├── wizard_of_oz.txt       # Primary literary training corpus (~232 KB)
├── output_train.txt       # Extracted OpenWebText training split (~385 MB)
├── output_val.txt         # Extracted OpenWebText validation split (~41 MB)
├── vocab.txt              # Character vocabulary file (one char per line)
│
├── requirements.txt       # Python dependencies
├── .gitignore
├── LICENSE
│
└── Research/
    ├── Small_Language_model.ipynb   # Jupyter notebook: exploration & experiments
    ├── model-01.pkl                 # Pickled model checkpoint (~928 KB)
    └── model-01.pt                  # PyTorch state_dict checkpoint (~32 MB)
```

---

## 🛠 Setup & Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU recommended (NVIDIA with CUDA 11.8+ or 12.x)
- Or Apple Silicon Mac (MPS backend supported)

### 1. Clone the repository

```bash
git clone https://github.com/Jeevant010/Mini_Generative_Pretrained_Transformer.git
cd Mini_Generative_Pretrained_Transformer
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For CUDA 12.1 specifically (recommended):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Dependencies:**

| Package | Purpose |
|---|---|
| `torch` | Core deep learning framework |
| `torchvision` | (Indirect dep, pulled alongside torch) |
| `torchaudio` | (Indirect dep, pulled alongside torch) |
| `transformers` | Tokenizer utilities (future BPE expansion) |
| `tokenizers` | Fast tokenizer backend |
| `datasets` | Dataset loading utilities |
| `tqdm` | Progress bars in `extract.py` |

---

## 🚀 Usage

### Step 1 — Data Extraction

> Skip this step if you want to train only on `wizard_of_oz.txt`.

Download the [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) dataset (or any `.xz`-compressed text corpus) and update the path in `extract.py`:

```python
folder_path = r"C:\path\to\your\openwebtext"
```

Then run:

```bash
python extract.py
```

This will create:
- `output_train.txt` — training corpus
- `output_val.txt` — validation corpus
- `vocab.txt` — character vocabulary

---

### Step 2 — Training

Ensure your `vocab.txt` exists (from Step 1 or built manually), and that `vocab_size` is correctly set in `training.py`. Then:

```bash
python training.py
```

Training progress is printed every 100 steps:

```
Using device: cuda
✅ Using torch.compile() optimized mode.
Step     0 | train loss 4.824 | val loss 4.832
Step   100 | train loss 3.211 | val loss 3.219
...
Step 19999 | train loss 1.573 | val loss 1.612
✅ Training complete. Final loss: 1.5728
💾 Model saved to model-01.pt
```

---

### Step 3 — Inference

Edit `main.py` to load the model checkpoint and provide a prompt:

```python
prompt = "Hello! Where the hell are you?"
context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=100)[0].tolist())
print(generated_chars)
```

Then run:

```bash
python main.py
```

You can also explore generation interactively in the Research notebook.

---

## 📓 Research Notebook

**`Research/Small_Language_model.ipynb`** — A Jupyter notebook containing:

- Step-by-step character-level tokenization experiments
- Proof-of-concept bigram language model baseline
- Iterative construction of the self-attention mechanism
- Loss curves and exploratory analysis
- A pickled model artifact (`model-01.pkl`) for quick reload

This notebook serves as the **learning scaffold** from which `training.py` was derived and is recommended reading for anyone wanting to understand the design decisions from first principles.

---

## 💾 Saved Artifacts

| File | Format | Size | Notes |
|---|---|---|---|
| `Research/model-01.pt` | `torch.save(state_dict)` | ~32 MB | Full trained weights, load with `model.load_state_dict(torch.load(...))` |
| `Research/model-01.pkl` | Python `pickle` | ~928 KB | Pickled model object from notebook experiments |

To load the `.pt` checkpoint:

```python
model = GPTLanguageModel(vocab_size)
model.load_state_dict(torch.load("Research/model-01.pt", map_location=device))
model.eval()
```

---

## 🖥 GPU / Device Support

The code automatically selects the best available device:

```python
device = (
    "cuda"  if torch.cuda.is_available()  else
    "mps"   if torch.backends.mps.is_available() else
    "cpu"
)
```

| Device | Notes |
|---|---|
| **CUDA (NVIDIA)** | Fully supported. AMP + `torch.compile` active. Fastest. |
| **MPS (Apple Silicon)** | Supported. AMP disabled (`float16` on MPS is partially supported). `torch.compile` may be limited. |
| **CPU** | Supported. No AMP or compile optimizations. Training will be very slow. |

---

## 🔮 Future Work & Improvements

- [ ] **BPE / WordPiece tokenization** — Replace character-level tokenizer with a learned subword tokenizer (e.g., via HuggingFace `tokenizers`) to dramatically reduce sequence length and improve semantic compression
- [ ] **Learning rate scheduler** — Add cosine annealing with warm-up (standard for GPT training)
- [ ] **Flash Attention** — Replace the manual scaled dot-product with `F.scaled_dot_product_attention` (PyTorch 2.0+) for fused, memory-efficient attention
- [ ] **Larger dataset** — Train on the full OpenWebText (not just 1% sample) with multi-GPU data parallelism (`torch.nn.DataParallel` or `DistributedDataParallel`)
- [ ] **Checkpoint resumption** — Save optimizer state alongside model weights so training can resume from any step
- [ ] **Evaluation metrics** — Add perplexity curves and sample logging to TensorBoard or Weights & Biases
- [ ] **Gradio / Streamlit demo** — Wrap the inference in a simple web UI for interactive text generation

---

## 📚 References

- Vaswani et al., *Attention Is All You Need* (2017) — [[arXiv:1706.03762]](https://arxiv.org/abs/1706.03762)
- Radford et al., *Improving Language Understanding by Generative Pre-Training* (GPT-1, 2018) — [OpenAI Blog](https://openai.com/research/language-unsupervised)
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019) — [OpenAI Blog](https://openai.com/research/better-language-models)
- Andrej Karpathy, *Let's build GPT: from scratch, in code, spelled out* — [YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY) | [nanoGPT](https://github.com/karpathy/nanoGPT)
- PyTorch Documentation — [pytorch.org](https://pytorch.org/docs/stable/index.html)
- OpenWebText Corpus — [skylion007.github.io](https://skylion007.github.io/OpenWebTextCorpus/)

---

<div align="center">

Made with ❤️ and gradient descent &nbsp;|&nbsp; MIT License

</div>