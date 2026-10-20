
import json, numpy as np, sys
sys.path.insert(0,'.')
from tokenizers import Tokenizer
from gpt2_numpy import load_safetensors, GPT2, softmax, D
tok = Tokenizer.from_file(f'{D}/tokenizer.json')
m = GPT2(load_safetensors(f'{D}/model.safetensors'))

def pack(prompt):
    ids = tok.encode(prompt).ids
    lg, A = m.forward(ids)
    n = len(ids); p = softmax(lg[-1]); top = np.argsort(-p)[:8]
    attn = [[[[round(float(A[l,h,i,j]),4) for j in range(n)] for i in range(n)] for h in range(12)] for l in range(12)]
    prof=[]
    for l in range(12):
        for h in range(12):
            M=A[l,h]
            prof.append([round(float(np.mean([M[i,0] for i in range(1,n)])),3),
                         round(float(np.mean([M[i,i-1] for i in range(1,n)])),3),
                         round(float(np.mean([M[i,i] for i in range(n)])),3),
                         round(float(np.mean([-(r*np.log(r+1e-12)).sum() for r in M])),3)])
    return {'prompt':prompt,'tokens':[tok.decode([i]) for i in ids],
            'next':[[tok.decode([int(i)]),round(float(p[i]),5)] for i in top],
            'layerSink':[round(float(np.mean([A[l,h,i,0] for h in range(12) for i in range(1,n)])),3) for l in range(12)],
            'prof':prof,'attn':attn}

out = 'const REAL_ATTN = {\n'
for key, pr in (('p5','GPT is a language'), ('p6','GPT is a language that')):
    b = pack(pr)
    a = b['attn']
    lines=[]
    for L in range(12):
        heads=[]
        for H in range(12):
            rows=','.join('['+','.join('%.4f'%v for v in row)+']' for row in a[L][H])
            heads.append('['+rows+']')
        lines.append('['+','.join(heads)+']')
    out += '%s:{"prompt":%s,"tokens":%s,"layerSink":[%s],"prof":[%s],\n"attn":[%s]},\n' % (
        key, json.dumps(pr), json.dumps(b['tokens']),
        ','.join(str(v) for v in b['layerSink']),
        ','.join('['+','.join(str(v) for v in x)+']' for x in b['prof']),
        ','.join(lines))
    print(key, b['tokens'], 'next0=', b['next'][0], 'L7sink=', b['layerSink'][7])
out = out.rstrip(',\n') + '\n};\n'
open('build_real_attn2.js','w',encoding='utf-8').write(out)
print('bytes', len(out))
