Run `python scripts/evaluate.py` from the repository root to regenerate:

- `metrics.json` and `metrics.csv`: Accuracy, word Precision/Recall/F1, OOV
  Recall and inference time for each available model.
- `error_analysis.csv`: sentence-level gold/prediction differences.
- `figures/model_comparison.png`: score comparison chart.

The script evaluates the whitespace and regex baselines. It includes the LSTM
only when both `checkpoints/lstm/best_lstm_model.pt` and
`checkpoints/lstm/vocab.json` are present. Those files are produced by the
separate LSTM training workflow. `predictions.csv` is also generated locally
for inspection but is not tracked because it duplicates the test corpus.