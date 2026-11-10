# Interactive Handwritten Digit Recognition Web Application

A modern, responsive deep learning web application for handwritten digit recognition ($0 - 9$) powered by a trained PyTorch neural network.

The web application features an interactive HTML5 drawing canvas with multi-touch, mouse, and stylus support, real-time center-of-mass image preprocessing adhering strictly to the official MNIST specification, and an asynchronous Flask REST API delivering prediction confidence and full 10-class probability distributions.

---

## Architecture & Model Integration

The application reconstructs and strictly loads the user's trained weights from `model.pth`:

- **Network Type:** Fully Connected Neural Network (Multilayer Perceptron / MLP)
- **Input Dimension:** $28 \times 28 = 784$ grayscale pixels
- **Hidden Layer:** $128$ units with **ReLU** activation
- **Output Layer:** $10$ logits ($0 - 9$)
- **Total Parameters:** $101,770$ parameters
  - `1.weight`: `[128, 784]`
  - `1.bias`: `[128]`
  - `3.weight`: `[10, 128]`
  - `3.bias`: `[10]`

The model weights are loaded once upon Flask startup (`strict=True`, `weights_only=True`), cached in memory, and set to evaluation mode (`model.eval()`) for zero-overhead inference with `torch.no_grad()`.

---

## Image Preprocessing Pipeline

The drawing input undergoes an identical preprocessing procedure to the original MNIST dataset:

1. **Decoding:** Decodes Base64 data URLs from the canvas into PIL Grayscale (`L`) images.
2. **Polarity Check:** Automatically detects and converts drawings to white digit strokes on a black background.
3. **Empty Canvas Detection:** Validates active stroke pixels; rejects empty or noisy drawings with an informative HTTP 400 error.
4. **Bounding Box Cropping:** Crops tightly around the handwritten digit.
5. **Aspect-Ratio Preserved Scaling:** Scales the digit so its largest dimension is $20$ pixels inside a $20 \times 20$ box.
6. **Center of Mass Translation:** Computes the intensity center-of-mass $(\bar{x}, \bar{y})$ and positions the digit onto a $28 \times 28$ canvas centered exactly at $(14, 14)$.
7. **Normalization:** Scales pixel values to $[0.0, 1.0]$ and standardizes using standard MNIST constants:
   $$\text{pixel}_{\text{norm}} = \frac{\text{pixel} - 0.1307}{0.3081}$$
8. **Tensor Conversion:** Reshapes to shape `[1, 1, 28, 28]` with `torch.float32`.

---

## Project Structure

```text
Interactive Handwritten Digit Recognition Web Application/
│
├── app.py                      # Flask REST API and web application server
├── model.pth                   # Trained PyTorch model checkpoint (original weights)
│
├── model/
│   ├── __init__.py             # Model package exports
│   └── cnn.py                  # Reconstructed PyTorch architecture & loader
│
├── utils/
│   ├── __init__.py             # Preprocessing exports
│   └── preprocessing.py        # MNIST-standard centering & normalization pipeline
│
├── templates/
│   └── index.html              # HTML5 responsive user interface
│
├── static/
│   ├── css/
│   │   └── style.css           # Modern dark-mode styling with animations
│   └── js/
│       └── script.js           # Drawing canvas engine & REST API consumer
│
├── verify_model.py             # 7-step model compatibility & checkpoint verifier
├── test_api.py                 # End-to-end Flask REST API test suite
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation & setup guide
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10+ (tested with Python 3.14)
- Pip

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Model Verification

Before launching the web server, you can verify model checkpoint integrity and compatibility:

```bash
python verify_model.py
```

This verifies:
1. Checkpoint location and loading.
2. Exact parameter shape matching across all 4 weight/bias tensors.
3. Strict model loading with zero missing/unexpected keys.
4. Architecture reconstruction: `Flatten -> Linear(784, 128) -> ReLU -> Linear(128, 10)`.
5. Inference with dummy input `[1, 1, 28, 28]`.
6. Output shape validation `[1, 10]`.
7. Softmax sum validation ($\sum p_i = 1.0$).

---

## Running the Application

### Option A: Development Server (Flask Built-in)
```bash
python app.py
```

### Option B: Production WSGI Server (Gunicorn)
```bash
gunicorn -w 2 -b 127.0.0.1:5000 app:app
```

The application will start on:
```text
http://127.0.0.1:5000
```

### Accessing the Web Application
Open your web browser and navigate to:
```text
http://127.0.0.1:5000
```

1. **Light / Dark Mode:** Use the theme toggle button in the top right to switch between Dark and Light mode (your preference is automatically remembered).
2. **Draw a Digit:** Draw any digit ($0 - 9$) on the black canvas using your mouse, touchscreen, or stylus.
3. **Brush Controls:** Select brush thickness (**Thin**, **Medium**, or **Thick**) if desired.
4. **Predict:** Click **Predict Digit** to trigger inference.
5. **Inspect Output:** Review the predicted digit, confidence percentage, the normalized $28 \times 28$ input preview, and the interactive 10-class probability distribution bar chart.
6. **Reset:** Click **Clear Canvas** to reset and draw a new digit.

---

## API Reference

### `POST /predict`
Submits an image for classification.

**Request Body (JSON):**
```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
}
```

**Response (JSON):**
```json
{
  "success": true,
  "prediction": 7,
  "confidence": 99.85,
  "probabilities": {
    "0": 0.000001,
    "1": 0.000045,
    "2": 0.000012,
    "3": 0.000100,
    "4": 0.000003,
    "5": 0.000021,
    "6": 0.000001,
    "7": 0.998500,
    "8": 0.000010,
    "9": 0.001307
  },
  "probabilities_list": [
    { "digit": 0, "probability": 0.000001, "percentage": "0.00%" },
    ...
    { "digit": 7, "probability": 0.998500, "percentage": "99.85%" }
  ],
  "thumbnail": "data:image/png;base64,iVBOR...",
  "message": "Digit successfully recognized as 7."
}
```

### `GET /info`
Returns model metadata, parameter counts, and loaded architecture summary.

---

## Automated Testing

Run the integration test suite:

```bash
python -m unittest test_api.py -v
```

