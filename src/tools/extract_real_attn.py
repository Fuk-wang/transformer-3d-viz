"""导出真实 GPT-2 small 的注意力权重 + 下一词分布；并检验站点那套"四类头"说法是否成立。"""
import json
import numpy as np
from tokenizers import Tokenizer
from gpt2_numpy import load_safetensors, GPT2, softmax, D

tok = Tokenizer.from_file(f'{D}/tokenizer.json')
m = GPT2(load_safetensors(f'{D}/model.safetensors'))


# ---------------- 校验 1：语言模型损失 ----------------
TEXT = ("The quick brown fox jumps over the lazy dog. Language models are trained to predict "
        "the next token in a sequence, one token at a time, using only the tokens that came "
        "before it. This simple objective, scaled up to billions of parameters and trillions "
        "of tokens, turns out to produce systems that can translate, summarize, and reason.")
ids = tok.encode(TEXT).ids
lg, _ = m.forward(ids, want_attn=False)
lg = lg[:-1]
tgt = np.array(ids[1:])
logp = lg - lg.max(-1, keepdims=True)
logp = logp - np.log(np.exp(logp).sum(-1, keepdims=True))
nll = -logp[np.arange(len(tgt)), tgt]
print(f'[校验] {len(ids)} tokens  平均 NLL = {nll.mean():.4f}  '
      f'困惑度 = {np.exp(nll.mean()):.2f}')
print('  （GPT-2 small 在通顺英文上 PPL 应在 20-40；若实现有错会飙到数百上千）')


# ---------------- 校验 2：真实下一词分布 ----------------
def next_dist(prompt, k=8):
    i = tok.encode(prompt).ids
    lg, A = m.forward(i)
    p = softmax(lg[-1])
    top = np.argsort(-p)[:k]
    return i, A, [(tok.decode([int(t)]), float(p[t])) for t in top]


print('\n[真实下一词]')
for pr in ['GPT is a language', 'The capital of France is']:
    i, _, dist = next_dist(pr)
    print(f'  {pr!r}  ({len(i)} tokens)')
    for s, p in dist[:5]:
        print(f'      {s!r:16s} {p:7.4f}')


# ---------------- 主交付：5 token 提示词的真实 12 头注意力 ----------------
PROMPT = 'GPT is a language'
ids5, A5, dist5 = next_dist(PROMPT, k=12)
toks5 = [tok.decode([t]) for t in ids5]
print(f'\n[交付] {PROMPT!r} -> {len(ids5)} tokens: {toks5}')
print(f'  注意力张量 shape = {A5.shape}  (层, 头, query, key)')
print(f'  行和范围 = [{A5.sum(-1).min():.6f}, {A5.sum(-1).max():.6f}]')
print(f'  因果掩码右上角 max = {max(A5[l, h][np.triu(np.ones((5,5), bool), 1)].max() for l in range(12) for h in range(12)):.1e}')

# 逐头行为统计（排除只能看自己的第 0 行）
print('\n[逐头行为] 各头把注意力给谁（排除第0行，n=4 个 query）')
print('  层  头  自己  前一个词  首词    其他   熵')
rows = []
for l in range(12):
    for h in range(12):
        a = A5[l, h]
        n = 5
        diag = np.mean([a[q, q] for q in range(1, n)])
        prev = np.mean([a[q, q - 1] for q in range(1, n)])
        first = np.mean([a[q, 0] for q in range(1, n)])
        other = np.mean([sum(a[q, j] for j in range(q - 1) if j != 0) for q in range(1, n)])
        ent = np.mean([-(a[q][:q + 1] * np.log(a[q][:q + 1] + 1e-12)).sum() for q in range(1, n)])
        rows.append((l, h, diag, prev, first, other, ent))
        print(f'  {l:2d} {h:3d}  {diag:.3f}  {prev:.3f}     {first:.3f}  {other:.3f}  {ent:.3f}')

# 汇总：各层"首词接收"强度
print('\n[按层] 第0个token（首词）接收到的平均注意力')
for l in range(12):
    v = np.mean([np.mean([A5[l, h][q, 0] for q in range(1, 5)]) for h in range(12)])
    bar = '#' * int(v * 60)
    print(f'  L{l:<2d} {v:.3f} {bar}')


# ---------------- 中文提示词的实际情况 ----------------
CN = 'GPT 是一种语言模型'
cn_ids = tok.encode(CN).ids
print(f'\n[中文提示词] {CN!r} -> {len(cn_ids)} tokens（不是 5）')
for t in cn_ids:
    print(f'    id={t:<6d} {tok.id_to_token(t)!r} -> {tok.decode([t])!r}')
lgc, Ac = m.forward(cn_ids)
print(f'  真实注意力矩阵是 {Ac.shape[2]}x{Ac.shape[3]}，不是 5x5')


# ---------------- 落盘 ----------------
out = {
    'model': 'gpt2 (124M, GPT-2 small)',
    'config': {'n_layer': 12, 'n_head': 12, 'd_model': 768, 'd_k': 64,
               'n_ff': 3072, 'vocab': 50257, 'n_ctx': 1024},
    'prompt': PROMPT,
    'tokens': toks5,
    'token_ids': ids5,
    'shape': [12, 12, len(ids5), len(ids5)],
    'layout': 'attention[layer][head][query][key]; key>query 的位置按因果掩码恒为 0',
    'row_sums': A5.sum(-1).round(6).tolist(),
    'attention': A5.round(5).tolist(),
    'next_token': [{'token': s, 'p': round(p, 6)} for s, p in dist5],
    'note_cn_prompt': {
        'text': CN,
        'n_tokens': len(cn_ids),
        'tokens': [tok.decode([t]) for t in cn_ids],
        'reason': 'GPT-2 是字节级 BPE，中文几乎不在词表里，每个汉字碎成 2-3 个字节 token',
    },
}
with open(f'{D}/gpt2-real-attn.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f'\n已写出 {D}/gpt2-real-attn.json')
