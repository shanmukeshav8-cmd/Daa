# Airport AI Assistant

A local Flask web app with an intent-based airport help chatbot, a sample-data flight-delay estimator, and a sample-data security queue estimator. The frontend uses HTML forms and CSS only; Flask handles form submissions and all chatbot and prediction logic. It does not connect to live airline, airport, weather, or booking systems.

## Requirements

- Python 3.10 or newer
- Packages in `requirements.txt`

## Run locally

In VS Code, open this project folder and select a Python 3.10 or newer interpreter. In the integrated PowerShell terminal, create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Train all three local models and start the web app:

```powershell
python train_chatbot.py
python train_delay.py
python train_queue.py
python app.py
```

Open http://127.0.0.1:5000. Models are saved in `models/`, which is created by the training scripts. The web page loads without trained models, but assistant and estimator requests return a clear message instructing you to train them first.

The assistant and both estimators submit standard HTML forms to Flask POST routes.

## Optional JSON API

- `GET /api/health` — reports whether all model files exist.
- `POST /api/chat` — JSON body: `{"message": "Where is baggage claim?"}`.
- `POST /api/predict/delay` — JSON fields: `airline`, `origin`, `destination`, `scheduled_hour` (0–23), `day_of_week` (Monday=0 through Sunday=6), `weather` (`clear`, `light_rain`, `fog`, or `storm`), and `traffic_level` (`low`, `medium`, or `high`).
- `POST /api/predict/queue` — JSON fields: `terminal` (1–3), `hour` (0–23), `day_of_week` (Monday=0 through Sunday=6), `passenger_volume` (20–300), and `open_lanes` (1–16).

Both prediction APIs validate their inputs and return estimates in minutes. The included CSVs are small illustrative training datasets; model results are demonstrations, not operational forecasts.

## Retraining

Edit the corresponding files in `data/` and rerun the matching `train_*.py` script. Each trainer checks the required columns and writes its model artifact under `models/`.

The `notebooks/eda.ipynb` notebook summarizes the sample data, while `notebooks/model_training.ipynb` runs the same training scripts.

## Tests

After training the models, run the API and validation tests:

```powershell
python -m unittest discover -s tests -v
```
