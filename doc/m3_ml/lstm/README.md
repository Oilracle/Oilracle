# QF5210: WTI LSTM forecasting contribution

Begin with [`START_HERE.md`](../../../m3_ml/lstm/START_HERE.md). Run the commands below from `m3_ml/lstm/`. The project includes executable code,
processed inputs, real trained models/results, an executed notebook, a detailed
technical report, editable slides, and a speaking/Q&A guide.

```bash
python -m pip install -r requirements.txt
python src/oil_lstm.py
python -m unittest discover -s tests -v
```

Use `python src/oil_lstm.py --forecast-only` to refresh from an updated processed
feature file. See the timing contract and limitations before comparing models.
