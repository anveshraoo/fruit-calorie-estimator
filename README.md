# 🍎 Smart Fruit Calorie Estimation using 3D Imaging & Machine Learning

A Streamlit-based web application that detects fruits, estimates their weight using image processing and 3D approximation, and calculates calorie content.

---

## 🚀 Features

* 🍌 Fruit detection using computer vision
* 📷 Supports image input (upload/webcam)
* 📐 Estimates fruit size using 3D approximation
* ⚖️ Converts volume → weight
* 🔥 Calculates calorie values
* 🌐 Interactive UI using Streamlit

---

## 🛠️ Tech Stack

* Python
* OpenCV
* NumPy
* Streamlit
* Machine Learning

---

## 📂 Project Structure

```
fruit-calorie-estimator/
│
├── app.py              # Main Streamlit application
├── model_new/          # Trained ML model
├── accuracy/           # Model evaluation results
├── dataset/            # Dataset (optional / may not be included)
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

### 1️⃣ Clone the repository

```bash
git clone https://github.com/anveshraoo/fruit-calorie-estimator.git
cd fruit-calorie-estimator
```

---

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run the application

```bash
streamlit run app.py
```

---

### 4️⃣ Open in browser

Streamlit will automatically open:

```
http://localhost:8501
```

---

## 🧠 How It Works

1. Capture or upload a fruit image
2. Detect fruit using image processing
3. Estimate dimensions using 3D approximation
4. Convert volume into weight
5. Calculate calories using predefined nutritional values

---

## 📊 Example Output

| Fruit  | Estimated Weight | Calories |
| ------ | ---------------- | -------- |
| Apple  | 150g             | 78 kcal  |
| Banana | 120g             | 105 kcal |

---

## 💼 Use Case

This project demonstrates practical implementation of:

* Computer Vision
* Machine Learning
* Real-time calorie estimation

Applications include:

* Diet tracking
* Smart health systems
* Fitness monitoring

---

## ⭐ Highlights

* End-to-end working system
* Combines AI + 3D estimation
* Real-time processing with Streamlit
* Practical healthcare application

---

## ⚠️ Dataset

The dataset may not be included in this repository due to size constraints.

---

## 📌 Note

This project provides approximate calorie values and is intended for educational purposes.

---

## 👨‍💻 Author

**Anvesh Rao**
GitHub: https://github.com/anveshraoo
