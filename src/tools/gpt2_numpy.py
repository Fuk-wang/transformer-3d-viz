"""纯 numpy 的 GPT-2 (124M) 前向推理 + 注意力权重导出。
不依赖 torch / transformers；权重从 safetensors 直接解析。
"""
import json, sys
import numpy as np

D = 'C:/Users/79305/WorkBuddy/2026-10-20-19-38-54/audit'
DT = {'F32': np.float32, 'F16': np.float16}


def load_safetensors(path):
    with open(path, 'rb') as f:
        n = int.from_bytes(f.read(8), 'little')
        hdr = json.loads(f.read(n).decode('utf-8'))
        base = 8 + n
        blob = f.read()
    out = {}
    for k, v in hdr.items():
        if k == '__metadata__':
            continue
        s, e = v['data_offsets']
        out[k] = np.frombuffer(blob[s:e], dtype=DT[v['dtype']]).reshape(v['shape'])
    return out


def ln(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def gelu_new(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class GPT2:
    def __init__(self, W, n_layer=12, n_head=12, n_embd=768):
        self.W, self.L, self.H, self.C = W, n_layer, n_head, n_embd
        self.dk = n_embd // n_head

    def forward(self, ids, want_attn=True):
        W = self.W
        ids = np.asarray(ids)
        n = len(ids)
        x = W['wte.weight'][ids] + W['wpe.weight'][:n]
        attns = []
        for i in range(self.L):
            p = f'h.{i}.'
            a = ln(x, W[p + 'ln_1.weight'], W[p + 'ln_1.bias'])
            qkv = a @ W[p + 'attn.c_attn.weight'] + W[p + 'attn.c_attn.bias']   # Conv1D: (in,out)
            q, k, v = np.split(qkv, 3, axis=-1)
            # (n, 12, 64) -> (12, n, 64)
            q = q.reshape(n, self.H, self.dk).transpose(1, 0, 2)
            k = k.reshape(n, self.H, self.dk).transpose(1, 0, 2)
            v = v.reshape(n, self.H, self.dk).transpose(1, 0, 2)
            s = q @ k.transpose(0, 2, 1) / np.sqrt(self.dk)                      # (12, n, n)
            mask = np.triu(np.ones((n, n), dtype=bool), 1)
            s = np.where(mask, -np.inf, s)
            A = softmax(s, axis=-1)
            attns.append(A)
            ctx = (A @ v).transpose(1, 0, 2).reshape(n, self.C)
            x = x + ctx @ W[p + 'attn.c_proj.weight'] + W[p + 'attn.c_proj.bias']
            m = ln(x, W[p + 'ln_2.weight'], W[p + 'ln_2.bias'])
            m = gelu_new(m @ W[p + 'mlp.c_fc.weight'] + W[p + 'mlp.c_fc.bias'])
            x = x + m @ W[p + 'mlp.c_proj.weight'] + W[p + 'mlp.c_proj.bias']
        x = ln(x, W['ln_f.weight'], W['ln_f.bias'])
        logits = x @ W['wte.weight'].T          # 权重绑定
        return logits, (np.stack(attns) if want_attn else None)


if __name__ == '__main__':
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f'{D}/tokenizer.json')
    W = load_safetensors(f'{D}/model.safetensors')
    print(f'张量数 {len(W)}  wte{W["wte.weight"].shape}  '
          f'c_attn{W["h.0.attn.c_attn.weight"].shape}  c_fc{W["h.0.mlp.c_fc.weight"].shape}')
    m = GPT2(W)

    # ---------- 正确性自检：GPT-2 对经典提示的补全 ----------
    print('\n[自检] 前向实现是否可信')
    for prompt, expect in [('The capital of France is', ' Paris'),
                           ('The Eiffel Tower is located in the city of', ' Paris')]:
        ids = tok.encode(prompt).ids
        lg, _ = m.forward(ids, want_attn=False)
        p = softmax(lg[-1])
        top = np.argsort(-p)[:3]
        names = [tok.decode([int(i)]) for i in top]
        hit = 'OK' if expect.strip().lower() == names[0].strip().lower() else 'MISS'
        print(f'  {hit:4s} {prompt!r}')
        print(f'       -> {[(repr(nm), round(float(p[i]),4)) for nm, i in zip(names, top)]}')
    sys.exit(0)
