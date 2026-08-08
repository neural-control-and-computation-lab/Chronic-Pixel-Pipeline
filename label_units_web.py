"""
Web labeler for unit classification training data.

Reads E:/labeling_pool/manifest.csv + figures, presents one unit at a time
in the browser, records labels to E:/labeling_pool/labels.csv.

Binary task: each unit is labeled 'neural' (real spikes) or 'noise' (artifact),
or 'skip' (uncertain — excluded from training).

Keyboard shortcuts (page must have focus):
  1   neural
  2   noise
  3   skip
  b   back (re-label previous)
  ←/→ navigate without labeling

Usage:
  python label_units_web.py
  # then open http://127.0.0.1:5000 in a browser
"""
import csv
import datetime as dt
import os
from pathlib import Path

from flask import (Flask, abort, jsonify, redirect, render_template_string,
                   request, send_from_directory, url_for)
import pandas as pd


DATA_ROOT = Path(os.environ.get('DATA_ROOT', 'E:/'))
POOL_DIR = DATA_ROOT / 'labeling_pool'
MANIFEST_PATH = POOL_DIR / 'manifest.csv'
LABELS_PATH = POOL_DIR / 'labels.csv'

app = Flask(__name__)


def load_manifest():
    return pd.read_csv(MANIFEST_PATH)


def load_labels():
    if not LABELS_PATH.exists():
        return pd.DataFrame(columns=['id', 'label', 'timestamp'])
    df = pd.read_csv(LABELS_PATH)
    if 'id' in df.columns:
        df['id'] = df['id'].astype(int)
    return df


def save_label(unit_id, label):
    """Append/overwrite a label row for unit_id."""
    df = load_labels()
    now = dt.datetime.now().isoformat(timespec='seconds')
    if unit_id in df['id'].values:
        df.loc[df['id'] == unit_id, 'label'] = label
        df.loc[df['id'] == unit_id, 'timestamp'] = now
    else:
        df = pd.concat([df, pd.DataFrame([{'id': unit_id,
                                            'label': label,
                                            'timestamp': now}])],
                        ignore_index=True)
    df = df.sort_values('id').reset_index(drop=True)
    df.to_csv(LABELS_PATH, index=False)


PAGE_HTML = """<!doctype html>
<html>
<head>
<title>Unit Labeler</title>
<style>
body { font-family: -apple-system, sans-serif; margin: 0;
       background: #1e1e1e; color: #eee; }
.bar { padding: 10px 20px; background: #2a2a2a;
       display: flex; justify-content: space-between; align-items: center;
       border-bottom: 1px solid #444; }
.meta { font-size: 14px; color: #aaa; }
.meta .pneural { color: #ffcc66; font-weight: bold; }
.btns { display: flex; gap: 8px; }
button { padding: 8px 16px; border: none; border-radius: 4px;
         color: white; font-weight: bold; cursor: pointer; font-size: 14px; }
.b-neural { background: #2c7a3a; }
.b-noise  { background: #a33636; }
.b-skip   { background: #555; }
.b-nav    { background: #4477aa; }
button:hover { filter: brightness(1.15); }
.figwrap { display: flex; justify-content: center; padding: 12px; }
img { max-width: 95vw; max-height: 80vh; border: 1px solid #444; }
.status { padding: 6px 20px; font-size: 13px; color: #99cc99;
          background: #2a2a2a; border-bottom: 1px solid #444; }
.progress { font-size: 13px; color: #ccc; }
.shortcut { color: #888; font-size: 12px; margin-left: 8px; }
.label-pill { padding: 2px 8px; border-radius: 10px; font-size: 12px;
              margin-left: 8px; }
.label-neural { background: #2c7a3a; color: white; }
.label-noise  { background: #a33636; color: white; }
.label-skip   { background: #555;    color: white; }
</style>
</head>
<body>
<div class="bar">
  <div class="meta">
    <strong>Unit {{ row.id }}</strong>
    &nbsp;|&nbsp; {{ row.source }} bank{{ '%02d'|format(row.bank) }}
    uid={{ row.unit_id }}
    &nbsp;|&nbsp; <span class="pneural">P(neural)={{ '%.2f'|format(row.p_neural) }}</span>
    &nbsp;|&nbsp; KS4={{ row.ks4_label }}
    {% if current_label %}
      <span class="label-pill label-{{ current_label }}">{{ current_label }}</span>
    {% endif %}
  </div>
  <div class="progress">
    {{ idx + 1 }} / {{ total }} &nbsp;|&nbsp;
    labeled: {{ n_labeled }} ({{ n_neural }} neural, {{ n_noise }} noise, {{ n_skip }} skip)
  </div>
</div>

<div class="bar">
  <div class="btns">
    <button class="b-neural" onclick="label('neural')">Neural<span class="shortcut">[1]</span></button>
    <button class="b-noise"  onclick="label('noise')">Noise<span class="shortcut">[2]</span></button>
    <button class="b-skip"   onclick="label('skip')">Skip<span class="shortcut">[3]</span></button>
  </div>
  <div class="btns">
    <button class="b-nav" onclick="go(-1)">◀ Prev<span class="shortcut">[←]</span></button>
    <button class="b-nav" onclick="go(1)">Next ▶<span class="shortcut">[→]</span></button>
  </div>
</div>

<div class="figwrap">
  <img src="{{ url_for('serve_image', unit_id=row.id) }}"
       alt="unit {{ row.id }}"/>
</div>

<script>
function label(lbl) {
  fetch('/api/label', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({id: {{ row.id }}, label: lbl})
  }).then(r => r.json()).then(_ => {
    window.location = '/?idx={{ idx + 1 }}';
  });
}
function go(delta) {
  let next = {{ idx }} + delta;
  if (next < 0) next = 0;
  window.location = '/?idx=' + next;
}
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === '1') label('neural');
  else if (e.key === '2') label('noise');
  else if (e.key === '3') label('skip');
  else if (e.key === 'ArrowLeft' || e.key === 'b') go(-1);
  else if (e.key === 'ArrowRight') go(1);
});
</script>
</body>
</html>
"""


@app.route('/')
def index():
    manifest = load_manifest()
    labels = load_labels()
    total = len(manifest)
    if total == 0:
        return "<h2>Pool is empty. Run build_labeling_pool.py first.</h2>"

    # Default: jump to first unlabeled
    idx_param = request.args.get('idx')
    if idx_param is None:
        labeled_ids = set(labels['id'].astype(int).tolist())
        unlabeled = manifest[~manifest['id'].isin(labeled_ids)]
        if len(unlabeled) > 0:
            idx = int(unlabeled.index[0])
        else:
            idx = 0
    else:
        idx = max(0, min(total - 1, int(idx_param)))

    row = manifest.iloc[idx].to_dict()
    current_label = None
    if int(row['id']) in labels['id'].values:
        current_label = labels.loc[labels['id'] == int(row['id']),
                                    'label'].iloc[0]

    n_labeled = len(labels)
    n_neural = (labels['label'] == 'neural').sum() if n_labeled else 0
    n_noise = (labels['label'] == 'noise').sum() if n_labeled else 0
    n_skip = (labels['label'] == 'skip').sum() if n_labeled else 0

    return render_template_string(
        PAGE_HTML, row=row, idx=idx, total=total,
        current_label=current_label,
        n_labeled=n_labeled, n_neural=int(n_neural),
        n_noise=int(n_noise), n_skip=int(n_skip))


@app.route('/img/<int:unit_id>')
def serve_image(unit_id):
    fname = f'unit_{unit_id:04d}.png'
    if not (POOL_DIR / fname).exists():
        abort(404)
    return send_from_directory(POOL_DIR, fname)


@app.route('/api/label', methods=['POST'])
def api_label():
    data = request.get_json(force=True)
    uid = int(data['id'])
    label = data['label']
    if label not in ('neural', 'noise', 'skip'):
        return jsonify({'error': 'bad label'}), 400
    save_label(uid, label)
    return jsonify({'ok': True})


@app.route('/api/stats')
def api_stats():
    labels = load_labels()
    out = {'total': int(len(labels))}
    for v in ('neural', 'noise', 'skip'):
        out[v] = int((labels['label'] == v).sum())
    return jsonify(out)


def main():
    if not MANIFEST_PATH.exists():
        print(f'Missing {MANIFEST_PATH}. Run build_labeling_pool.py first.')
        return
    print(f'Pool: {POOL_DIR}')
    print(f'Open http://127.0.0.1:5000  (Ctrl+C to stop)')
    app.run(host='127.0.0.1', port=5000, debug=False)


if __name__ == '__main__':
    main()
