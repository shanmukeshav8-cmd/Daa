# Airport AI Assistant — Runway

A local Flask web application providing an intent-based airport help chatbot plus two sample-data estimators: a flight-delay estimator and a security-queue estimator. The UI is a simple HTML/CSS frontend that posts forms to Flask; models are trained locally from the files in data/ and saved into models/ by the training scripts. The project is educational/demo-quality — the estimators use small sample datasets and are not live forecasts.

Stack / Technologies
- Language: Python 3.10+
- Framework: Flask
- Data + ML: pandas, scikit-learn, joblib
- Notebooks: Jupyter (not required to run app)

Key features
- Intent-based assistant (chatbot) driven by data/intents.json
- Flight-delay estimator trained on data/flight_delay.csv
- Security queue estimator trained on data/queue_data.csv
- Simple HTML web UI (templates/index.html) and optional JSON API
- Training scripts create models/ with trained sklearn pipelines

Project layout (ordered)

Daa-main/
├── README.md
├── .gitignore
├── requirements.txt
├── app.py
│
├── SRS/
│   └── README.md
│
├── data/
│   ├── flight_delay.csv
│   ├── intents.json
│   └── queue_data.csv
│
├── notebooks/
│   ├── eda.ipynb
│   └── model_training.ipynb
│
├── static/
│   └── styles.css
│
├── templates/
│   └── index.html
│
├── tests/
│   └── test_app.py
│
├── train_chatbot.py
├── train_delay.py
└── train_queue.py

Notes on the tree
- This tree matches the repository contents and the order you requested. No files were moved or renamed — this README simply reflects the ordered structure.
- Do NOT delete or move files: the application and training scripts reference data/, templates/, static/, and models/ relative to the repository root.
- The training scripts create a models/ directory at runtime. By default, models/*.pkl are ignored in .gitignore because trained model binaries are typically not committed.

Installation
1. Create and activate a virtual environment

Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS / Linux (bash)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

2. (Optional) If you plan to run the notebooks interactively, install Jupyter in the environment: `pip install jupyter`

Training models

The project ships with small sample datasets in data/. Each trainer validates required columns and writes a model artifact under models/.

```bash
python train_chatbot.py    # trains intent classifier using data/intents.json
python train_delay.py      # trains delay estimator using data/flight_delay.csv
python train_queue.py      # trains queue estimator using data/queue_data.csv
```

Starting the web app

After training (or even if you skip training — the UI will load but estimator/chat functionality requires models), start the Flask app:

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser. The UI shows a "Models ready" indicator when models/chatbot_model.pkl, models/delay_model.pkl and models/queue_model.pkl exist.

JSON API (optional)

- GET /api/health — reports which model files exist
- POST /api/chat — body JSON: {"message": "Where is baggage claim?"}
- POST /api/predict/delay — fields: airline, origin, destination, scheduled_hour (0–23), day_of_week (0–6), weather, traffic_level
- POST /api/predict/queue — fields: terminal (1–3), hour (0–23), day_of_week (0–6), passenger_volume (20–300), open_lanes (1–16)

All prediction APIs validate inputs and return estimates in minutes (see app.py for exact validation rules).

Tests

Run the test suite with unittest after training the models (tests assume the models exist):

```bash
python -m unittest discover -s tests -v
```

Dependencies check

The requirements.txt lists the packages used by the application and training scripts:
- Flask>=3.0,<4.0
- joblib>=1.3,<2.0
- pandas>=2.2,<4.0
- scikit-learn>=1.5,<2.0

This covers the imports visible in app.py and the train_*.py scripts. The scikit-learn installation will bring in numpy and scipy as needed. If you intend to run notebooks interactively, add Jupyter to your environment.

What I changed

- Updated README.md to present the repository tree in the exact ordered layout you supplied. No files were moved or modified other than README.md and .gitignore (the latter was updated earlier).

If you'd like next

- I can add a small GitHub Actions workflow to run tests on push (requires training steps or caching model artifacts).
- I can create a short CONTRIBUTING.md or a release checklist.
