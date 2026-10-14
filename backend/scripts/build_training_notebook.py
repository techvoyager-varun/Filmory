"""Build a standalone notebook embedding the reviewed training source."""
import ast
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
cells = []
def md(text): cells.append({'cell_type':'markdown','metadata':{},'source':text.splitlines(True)})
def code(text):
    ast.parse(text)
    cells.append({'cell_type':'code','metadata':{},'source':text.splitlines(True),'outputs':[],'execution_count':None})
md('''# Filmory: reproducible training and honest evaluation

This replaces the original notebook's model contracts and training loop. Use a GPU runtime for larger budgets. Low training loss alone does not establish recommendation accuracy. Checkpoints are selected by validation NDCG; sampled and full-catalogue test results remain separate. This notebook embeds the source and runs independently of the Filmory database and API keys.

Run all cells in order. Download the official MovieLens 20M archive from https://grouplens.org/datasets/movielens/20m/ and extract `ratings.csv` and `movies.csv` into the data directory below. Dataset access and terms remain those of GroupLens. No results are fabricated or prefilled.''')
code('''import sys, subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', 'numpy', 'pandas', 'matplotlib'])
''')
embedded = {
 'app/__init__.py':'', 'app/ml/__init__.py':'',
 'app/ml/architectures.py':(ROOT/'app/ml/architectures.py').read_text(encoding='utf-8'),
 'scripts/train_models.py':(ROOT/'scripts/train_models.py').read_text(encoding='utf-8'),
 'scripts/prepare_training_data.py':(ROOT/'scripts/prepare_training_data.py').read_text(encoding='utf-8'),
 'scripts/compare_finetuning.py':(ROOT/'scripts/compare_finetuning.py').read_text(encoding='utf-8'),
}
code('from pathlib import Path\nimport json, datetime\nworkspace = Path.cwd() / "filmory_training"\nfiles = '+repr(embedded)+'''\nfor relative, source in files.items():
    path = workspace / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding='utf-8')
data_dir = Path.cwd() / 'ml-20m'  # change to your extracted CSV directory
stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
source_dir = workspace / 'data' / stamp
run_dir = workspace / 'training' / 'runs' / stamp
''')
md('''## Prepare and split before fitting
Select 30% of users with seed 42. Sort ratings by timestamp and break ties by movie ID. The final two interactions become validation and test targets before fitting, and genre profiles are rebuilt from training interactions only. Both targets are scored with the same static training prefix. Mappings use the selected transductive catalogue; this does not evaluate unseen-item cold start.''')
code('''assert (data_dir/'ratings.csv').exists() and (data_dir/'movies.csv').exists(), 'Set data_dir to the extracted MovieLens CSV directory'
subprocess.check_call([sys.executable, str(workspace/'scripts/prepare_training_data.py'),
    '--ratings-csv', str(data_dir/'ratings.csv'), '--movies-csv', str(data_dir/'movies.csv'), '--output', str(source_dir)])
''')
md('''## Train the baseline and two learned components
The hybrid starts from the baseline fitted in this clean run. The Transformer samples users uniformly, uses a dedicated padding index, and pools the last valid sequence position. Training uses stable logit BCE plus a pairwise ranking term, gradient clipping, learning-rate reduction, and validation checkpoint selection. Increase budgets only using validation results; reserve untouched test users for the final experiment.''')
code('''subprocess.check_call([sys.executable, str(workspace/'scripts/train_models.py'),
    '--source', str(source_dir), '--output', str(run_dir), '--epochs', '10', '--sequence-epochs', '5',
    '--positives-per-epoch', '1500000', '--sequence-samples', '120000', '--embedding-dim', '16',
    '--eval-users', '400', '--catalogue-users', '200', '--threads', '4'])
''')
code('''metrics = json.loads((run_dir/'metrics.json').read_text())
for name in ['sampled_test', 'full_catalogue_validation', 'full_catalogue_test']:
    print(name, json.dumps(metrics[name], indent=2))
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, records in metrics['epochs'].items():
    axes[0].plot([r['epoch'] for r in records], [r['bce'] for r in records], marker='o', label=name)
    axes[1].plot([r['epoch'] for r in records], [r['validation']['ndcg10'] for r in records], marker='o', label=name)
axes[0].set(xlabel='Epoch', ylabel='Training BCE')
axes[1].set(xlabel='Epoch', ylabel='Sampled validation NDCG@10')
for ax in axes: ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(run_dir/'training_curves.png', dpi=160)
plt.show()
''')
md('''## Optional smaller-learning-rate continuation
Set the flag below only when you want an additional experiment. It keeps the clean training histories, preserves the parent whenever validation NDCG does not improve, and uses a new test-user population. The fixed-example comparison reports training-probe and validation BCE under the same one-positive/four-negative ratio. Do not repeatedly tune on the new test results.''')
code('''DO_FINE_TUNE = False
if DO_FINE_TUNE:
    fine_dir = run_dir.parent / (stamp + '_fine_tune')
    subprocess.check_call([sys.executable, str(workspace/'scripts/train_models.py'),
        '--source', str(source_dir), '--continue-from', str(run_dir), '--output', str(fine_dir),
        '--embedding-dim', '16', '--epochs', '5', '--sequence-epochs', '5',
        '--learning-rate', '0.0002', '--positives-per-epoch', '1500000', '--sequence-samples', '120000',
        '--test-user-offset', '800', '--eval-users', '400', '--catalogue-users', '200'])
    subprocess.check_call([sys.executable, str(workspace/'scripts/compare_finetuning.py'),
        '--parent', str(run_dir), '--child', str(fine_dir)])
''')
md('''## Reporting and reproducibility
Use `metrics.json`, `training_log.jsonl`, `split.json`, `provenance.json`, candidate arrays, and source snapshots as evidence. Report sample sizes, negative sampling, objective, training budgets, seed and static-prefix protocol. A hit rate against 99 negatives is not overall prediction accuracy. Repeat multiple seeds, increase catalogue evaluation users, assess explanation claims independently, and run a new blinded assistant set before making publication claims. Turnitin and journal acceptance have not been verified by this notebook.''')
notebook = {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'cells':cells}
for i, cell in enumerate(cells): cell['id'] = f'filmory-{i}'
output = ROOT/'training/filmory_training_v2.ipynb'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(notebook, indent=1), encoding='utf-8')
print(output)
