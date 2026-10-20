#!/bin/bash
# 把 build/v3-* 拼成单文件，然后用本机无头 Chrome 做启动自检 + 出图
set -e
cd "$(dirname "$0")"
OUT="../index.html"
cat v3-1.html v3-2.html v5-real.html v3-3.html v3-4.html v3-4b.html v4-stage.html v3-6.html > "$OUT"
echo "built: $(wc -c < "$OUT") bytes, $(grep -c '' "$OUT") lines"

CH="/c/Program Files/Google/Chrome/Application/chrome.exe"
URL="file:///C:/Users/79305/Desktop/WebTasks/04-transformer-3d-viz/transformer-3d-viz/index.html"
DOM="$TMPDIR/v3dom.html"
"$CH" --headless=new --disable-gpu --enable-unsafe-swiftshader --virtual-time-budget=5000 \
  --window-size=1600,900 --dump-dom "$URL?stage=4" > "$DOM" 2>"$TMPDIR/v3dom.err" || true

python - "$(cygpath -w "$DOM")" <<'PY'
import re,sys,json
s=open(sys.argv[1],encoding='utf-8',errors='replace').read()
m=re.search(r'<div id="fatal"[^>]*>(.*?)</div>',s,re.S)
bad = m and 'class="on"' in m.group(0)
print('FATAL:', (m.group(1)[:500] if bad else 'none'))
m=re.search(r'id="diag"[^>]*>(.*?)</pre>',s,re.S)
if not m:
    print('DIAG: MISSING'); sys.exit(1)
d=json.loads(m.group(1))
print('DIAG keys:', list(d.keys()))
print(' mode/gates:', d['mode'], d['gates'], d['gateGap'])
print(' blocks/heads:', d['blocks'], d['heads'], 'renderer:', d['renderer'][:48])
print(' attn rowsums:', d['attn_rowsums'])
print(' mask all-zero-above:', d['masked_upper_zero'])
print(' beadPos:', d['beadPos'])
print(' labels:', s.count('class="lbl'), 'stageBtns:', s.count('class="n"'))
PY

# 截图：框图 / 解剖 / 解剖 2D / 总览
mkdir -p ../shots-v3
shot () {
  local name="$1"; shift
  local qs="$1"
  "$CH" --headless=new --disable-gpu --enable-unsafe-swiftshader --hide-scrollbars \
    --virtual-time-budget=6000 --run-all-compositor-stages-before-draw \
    --window-size=1600,900 --screenshot="$(cygpath -w ../shots-v3/$name)" "$URL$qs" >/dev/null 2>&1 || true
  echo "shot: $name  $(ls -la ../shots-v3/$name | awk '{print $5}') bytes"
}
shot box-attention.png   "?stage=4"
shot box-overview.png    "?stage=2"
shot anat-attention.png  "?mode=anatomy&stage=4"
shot anat-2d.png         "?mode=anatomy&stage=8&view=2d"
shot anat-top.png        "?mode=anatomy&stage=10"
shot anat-query.png      "?mode=anatomy&stage=5&q=1&head=1"
shot box-2d.png          "?stage=4&view=2d"
