# Facial Emotion Recognition with CNN

Facial emotion detection using Convolutional Neural Networks (CNN) built with TensorFlow and Keras. This model classifies grayscale facial images into 7 distinct emotional categories, with preprocessing, training, and model saving included.

**📦 Repository:** [https://github.com/ahsankhizar5/facial-emotion-recognition-cnn](https://github.com/ahsankhizar5/facial-emotion-recognition-cnn)

---

## ✨ Features

* **7-Emotion Classification:** Detects happy, sad, angry, surprised, neutral, fearful, and disgusted expressions.
* **CNN Architecture:** Convolutional layers with dropout and dense layers for robust classification.
* **Image Preprocessing:** Uses `ImageDataGenerator` for data scaling and flow.
* **Model Persistence:** Trained model saved as `emotion_model.h5` for reuse and deployment.
* **Organized Dataset Structure:** Structured into `train`, `validation`, and `test` folders.

---

## 🚀 Getting Started

### 1. **Clone the Repository**

```bash
git clone https://github.com/ahsankhizar5/facial-emotion-recognition-cnn.git
cd facial-emotion-recognition-cnn
````

### 2. **Install Dependencies**

```bash
pip install tensorflow
```

> Optionally, set up a virtual environment for clean dependency management.

### 3. **Prepare Your Dataset**

Ensure your dataset is structured like:

```
data/
├── train/
│   ├── angry/
│   ├── happy/
│   └── ...
├── validation/
│   ├── sad/
│   ├── neutral/
│   └── ...
```

### 4. **Train the Model**

```bash
python train_emotion_model.py
```

This will train the model and save it as `emotion_model.h5`.

---

## 🛠️ Tech Stack

* **Python**
* **TensorFlow & Keras**
* **CNN**
* **ImageDataGenerator**

---

## 📁 Folder Structure

```
.
├── data/
│   ├── train/
│   └── validation/
├── saved_predictions/
├── emotion_model.h5
├── emotion_gui_app.py
├── train_emotion_model.py
└── README.md
```

---

## 🤝 Want to Contribute?

1. **Fork the repo**

2. **Create a branch**

   ```bash
   git checkout -b feature/your-feature
   ```

3. **Commit your changes**

   ```bash
   git add .
   git commit -m "Add your feature"
   ```

4. **Push and submit a PR**

   ```bash
   git push origin feature/your-feature
   ```

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🌟 Give a Star

If you find this project helpful or insightful, please **give it a ⭐ on GitHub** — it helps others discover it too!

---

> 😊 *"Machines can learn to recognize our emotions — let’s train them with care."*
