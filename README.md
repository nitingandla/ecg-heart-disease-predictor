# ECG Heart Disease Predictor

A deep learning web app that analyzes ECG images and flags potential cardiac conditions — including Myocardial Infarction, abnormal heartbeats, and MI history. Upload a standard 12-lead ECG scan and get a diagnosis in seconds.

**Live demo → [ecg-heart-disease-predicto.onrender.com](https://ecg-heart-disease-predicto.onrender.com)**

---

## What it does

You upload an ECG image. The app splits it into individual lead regions, runs each through a trained CNN, and outputs one of four diagnoses:

- Myocardial Infarction
- History of MI
- Abnormal Heartbeat
- Normal ECG

The thresholds are tuned from the model's softmax outputs rather than a simple argmax, so borderline cases lean toward caution — which matters in a medical context.

---

## How the model was built

The backbone is **MobileNetV2** pretrained on ImageNet, fine-tuned for ECG classification. Training used a 4-class dataset organized by condition, with an 80/20 train-validation split.

```
MobileNetV2 (frozen base)
→ GlobalAveragePooling2D
→ Dense(128, relu)
→ Dropout(0.4)
→ Dense(4, softmax)
```

Trained for 15 epochs with Adam at lr=0.0001, categorical crossentropy loss. Input images are resized to 224×224 and normalized to [0, 1].

The final model was exported as `.keras`, then converted to **ONNX format** for deployment — this removes the TensorFlow runtime dependency entirely and makes the app portable across Python versions.

---

## Repo structure

```
ecg-heart-disease-predictor/
│
├── app.py                  # Streamlit UI + inference logic
├── train_1d.py             # Model training script (MobileNetV2)
├── model.py                # Baseline model (PCA + LogisticRegression)
├── preprocessing.py        # Image preprocessing utilities
├── lead_extraction.py      # ECG lead grid extraction
├── signal_extraction.py    # Signal-level processing
├── signal_processing.py    # Signal filtering and cleanup
├── prepare_signal_data.py  # Dataset preparation
├── split_data.py           # Train/val/test splitting
│
├── ecg_best.keras          # Trained Keras model
├── class_indices.json      # Class label mapping
└── requirements.txt        # Python dependencies
```

---

## Tech stack

| Layer              |          Tool                   |
|--------------------|---------------------------------|
| Model architecture | MobileNetV2 (transfer learning) |
| Training framework | TensorFlow / Keras              |
| Model serving      | ONNX Runtime                    |
| Image processing   | OpenCV, Pillow                  |
| Web framework      | Streamlit                       |
| Language           | Python 3.10                     |
| Deployment         | Render                          |

---

## Deployment

The app is deployed on **Render** as a web service.

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

Render was chosen after hitting Python version compatibility walls on Streamlit Cloud — TensorFlow and most ML packages don't have wheels for Python 3.13+ yet. Render gives a proper Linux environment where the runtime is predictable.

Note: the free tier spins down after inactivity, so the first load after a cold start takes about 30–60 seconds. That's normal.

---

## Disclaimer

This is a personal project built for learning purposes. It is not a medical device and should not be used for actual clinical decision-making. Always consult a qualified cardiologist for ECG interpretation.

---

## Author

**Nitin Gandla** — [github.com/nitingandla](https://github.com/nitingandla)
