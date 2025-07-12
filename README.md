# 🧠 KMNIST CNN Classifier

A simple convolutional neural network built with PyTorch to classify Japanese Hiragana characters from the KMNIST dataset. This project demonstrates basic image classification on a non-Latin dataset using a custom CNN.

---

## 📌 Project Overview

- 🔤 10-class classification task (Hiragana characters)  
- 📈 Trained with PyTorch on CPU  
- 🧪 Evaluates accuracy and visualizes correct vs. incorrect predictions  
- 🖼️ Uses Matplotlib to plot prediction results

---

## 🧱 Model Summary

The CNN model is defined in `model.py` and includes:

- 5 convolutional layers with ReLU activation  
- Max pooling after each conv layer  
- Fully connected layers with ReLU  
- `nn.LazyLinear` used for dynamic input sizing

Example:

```python
model = CNN(input_layer=1, hidden_layer=200, output_layer=10)
```

---

## 🧪 Example Output

After training, a 5×5 grid shows predictions:

- ✅ Green titles: Correct predictions  
- ❌ Red titles: Incorrect predictions

Make sure to save your output in `Figure_1.png` if you'd like it rendered here.

![Prediction Grid](Figure_1.png)

---

## 🚀 Getting Started

### 1. Clone this repository

```bash
git clone https://github.com/your-username/kmnist-cnn-classifier.git
cd kmnist-cnn-classifier
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

On Windows, if you encounter SSL certificate issues with KMNIST, add this in your code:

```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

### 3. Run training

```bash
python main.py
```

---

## 🧾 Requirements

Listed in `requirements.txt`:

- torch>=2.0.0  
- torchvision>=0.15.0  
- matplotlib>=3.5.0

---

## 📁 Project Structure

```
├── main.py              # Training and evaluation loop  
├── model.py             # CNN model definition  
├── images/              # (Optional) folder for output plots  
├── data/                # Auto-downloaded KMNIST dataset  
├── requirements.txt     # Package dependencies  
└── README.md
```

---

## 🧠 Credits

- KMNIST dataset from ROIS-DS Center for Open Data in the Humanities  
- Built using PyTorch and Matplotlib

---

## 📜 License

This project is licensed under the MIT License.
