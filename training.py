

import argparse
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
block_size = 128
n_embd = 384
n_head = 1
n_layer = 1
dropout = 0.2
vocab_size = 0
string_to_int = {}
int_to_string = {}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Load a saved GPT-like model and generate text."
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=Path("vocab.txt"),
        help="Path to the vocabulary file",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("model.pkl"),
        help="Path to the saved model file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt to generate from. If omitted, interactive mode is used.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k sampling cutoff. Use 0 to disable.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=("cpu", "cuda", "mps"),
        help="Override the device selection",
    )
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--n-head", type=int, default=1)
    parser.add_argument("--n-layer", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    return parser


def configure_runtime(args):
    global block_size, n_embd, n_head, n_layer, dropout, device

    block_size = args.block_size
    n_embd = args.n_embd
    n_head = args.n_head
    n_layer = args.n_layer
    dropout = args.dropout

    if args.device is not None:
        requested = torch.device(args.device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if requested.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        device = requested


def load_vocab(vocab_path):
    global vocab_size, string_to_int, int_to_string

    chars = vocab_path.read_text(encoding="utf-8")
    if not chars:
        raise ValueError(f"Vocabulary file is empty: {vocab_path}")

    chars = sorted(set(chars))
    string_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_string = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)
    print(f"Using device: {device}")
    print(f"Loaded vocab from: {vocab_path}")
    print(f"vocab_size: {vocab_size}")


def encode(text):
    missing = sorted({ch for ch in text if ch not in string_to_int})
    if missing:
        preview = "".join(missing[:10])
        raise ValueError(
            f"Prompt contains characters that are not in the vocab: {preview!r}"
        )
    return [string_to_int[ch] for ch in text]


def decode(tokens):
    return "".join(int_to_string[token] for token in tokens)


class SingleHeadCausalSelfAttention(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        q = self.q_proj(x).unsqueeze(1)
        k = self.k_proj(x).unsqueeze(1)
        v = self.v_proj(x).unsqueeze(1)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=dropout if self.training else 0.0,
            is_causal=True,
        ).squeeze(1)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out


class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")

        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, channels = x.size()
        q = self.q_proj(x).view(
            batch_size, seq_len, self.n_head, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(x).view(
            batch_size, seq_len, self.n_head, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(x).view(
            batch_size, seq_len, self.n_head, self.head_dim
        ).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=dropout if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        if n_head > 1:
            self.impl = MultiHeadCausalSelfAttention(n_embd, n_head)
        else:
            self.impl = SingleHeadCausalSelfAttention(n_embd)

    def forward(self, x):
        return self.impl(x)


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = CausalSelfAttention(n_embd, n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.token_embedding_table.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        _, seq_len = idx.size()
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(seq_len, device=device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        batch_size, seq_len, channels = logits.size()
        loss = F.cross_entropy(
            logits.view(batch_size * seq_len, channels),
            targets.view(batch_size * seq_len),
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                limit = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, limit)
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx


def load_model(model_path):
    suffix = model_path.suffix.lower()
    if suffix in {".pt", ".pth", ".bin"}:
        payload = torch.load(model_path, map_location=device)
        if isinstance(payload, nn.Module):
            model = payload
        else:
            model = GPTLanguageModel()
            if isinstance(payload, dict) and "state_dict" in payload:
                payload = payload["state_dict"]
            model.load_state_dict(payload)
    else:
        with model_path.open("rb") as handle:
            model = pickle.load(handle)

    if not isinstance(model, nn.Module):
        raise TypeError(f"Loaded object is not a torch model: {type(model)!r}")

    model = model.to(device)
    model.eval()
    print(f"Loaded model from: {model_path}")
    return model


def generate_text(model, prompt, max_new_tokens, temperature, top_k):
    encoded_prompt = encode(prompt)
    context = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0)
    generated = model.generate(
        context,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    return decode(generated[0].tolist())


def main():
    args = build_parser().parse_args()
    configure_runtime(args)

    if not args.vocab_path.is_file():
        raise FileNotFoundError(f"Vocabulary file not found: {args.vocab_path}")
    if not args.model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    load_vocab(args.vocab_path)
    model = load_model(args.model_path)
    top_k = None if args.top_k <= 0 else args.top_k

    if args.prompt is not None:
        print(
            generate_text(
                model,
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=top_k,
            )
        )
        return

    while True:
        prompt = input("Enter prompt (or 'exit' to quit): ").strip()
        if prompt.lower() == "exit":
            break
        if not prompt:
            continue
        print(
            generate_text(
                model,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=top_k,
            )
        )


if __name__ == "__main__":
    main()
