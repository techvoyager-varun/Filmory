# Filmory training

`filmory_training_v2.ipynb` is the corrected standalone notebook. The supplied original is retained in `source_notebook.ipynb` for reviewed model-contract comparisons. Never run arbitrary notebook attachment code without review.

The corrected notebook is regenerated from the current reviewed Python sources with `scripts/build_training_notebook.py`. It has no saved experimental outputs and does not contain prefilled performance claims. Reviewing or regenerating it does not start training.

`source_notebook.ipynb` is an archival copy, with historical outputs and IPython shell commands (including a multiline ZIP command). Those cells are not plain Python scripts and are not evidence that the corrected pipeline was rerun. Use the corrected notebook for new experiments. The `20261001_finetune_low_lr` directory is an incomplete, inactive experiment with no completed metric report; its logs must not be represented as a successful training run.

Fresh training:
```powershell
cd E:/Filmory/backend
.venv/Scripts/python.exe scripts/train_models.py --output training/runs/NEW_RUN --embedding-dim 16 --epochs 10 --sequence-epochs 5
```

Lower-learning-rate continuation requires the SAME source histories and mappings as the parent. The script verifies the targets/history identity and refuses legacy checkpoints. Use a new output directory and an untouched test-user offset, keeping the parent validation population fixed:
```powershell
.venv/Scripts/python.exe scripts/train_models.py --output training/runs/NEW_CONTINUATION --continue-from training/runs/20261001_clean_extended --embedding-dim 16 --epochs 5 --sequence-epochs 5 --learning-rate 0.0002 --positives-per-epoch 1500000 --sequence-samples 120000 --test-user-offset 1200
.venv/Scripts/python.exe scripts/compare_finetuning.py --parent training/runs/20261001_clean_extended --child training/runs/NEW_CONTINUATION
.venv/Scripts/python.exe scripts/check_ranking_policy.py --bundle training/runs/NEW_CONTINUATION
```

The fresh-run source defaults to `backend/ml`; if the parent was trained using original CSVs or a prepared source, provide the same `--source` and `--ratings-csv` options when continuing. A parent checkpoint remains selected unless validation NDCG improves. Optimizer state is fresh; continuation is not an exact interrupted optimizer resume. Lower training loss by itself never approves deployment. Compare on identical validation data and reserve another untouched test set for future model development.

Metrics and source snapshots remain reviewable. Large datasets/checkpoints are local and ignored by Git; deployment must include the entire selected bundle. The active pointer is `backend/ml/active_bundle.json`. Original checkpoints remain in `backend/ml`.
