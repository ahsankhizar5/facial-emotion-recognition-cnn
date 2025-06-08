import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Load model and define labels
model = tf.keras.models.load_model("emotion_model.h5")
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Ensure directory exists to save processed images
os.makedirs("saved_predictions", exist_ok=True)

def preprocess_image(path):
    img = Image.open(path).convert('L')  # Grayscale
    img = np.array(img)

    # ✅ Histogram Equalization
    img_eq = cv2.equalizeHist(img.astype(np.uint8))

    img_resized = cv2.resize(img_eq, (48, 48))
    img_normalized = img_resized / 255.0
    return img_normalized.reshape(1, 48, 48, 1), img_resized

def predict_emotion():
    file_path = filedialog.askopenfilename(title="Select Facial Image")
    if file_path:
        img_array, processed_img = preprocess_image(file_path)
        preds = model.predict(img_array)[0]
        top3 = np.argsort(preds)[-3:][::-1]
        result = "\n".join([f"{class_labels[i]}: {round(preds[i]*100, 2)}%" for i in top3])
        result_text.set(result)

        # ✅ Save Processed Image with Predicted Label
        predicted_label = class_labels[top3[0]]
        save_path = f"saved_predictions/{predicted_label}_{os.path.basename(file_path)}"
        cv2.imwrite(save_path, processed_img * 255)

        # ✅ Save Chart
        plt.figure(figsize=(6, 4))
        plt.bar(class_labels, preds, color='skyblue')
        plt.title("Emotion Prediction")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("prediction_chart.png")
        plt.close()

        # Display original image in GUI
        img_display = Image.open(file_path).resize((150, 150))
        photo = ImageTk.PhotoImage(img_display)
        image_label.config(image=photo)
        image_label.image = photo

def apply_filter(filter_type):
    file_path = filedialog.askopenfilename(title="Select Image for Filtering")
    if file_path:
        img = cv2.imread(file_path)
        if filter_type == 'blur':
            img = cv2.GaussianBlur(img, (7, 7), 0)
        elif filter_type == 'sharpen':
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            img = cv2.filter2D(img, -1, kernel)
        elif filter_type == 'edge':
            img = cv2.Canny(img, 100, 200)
        img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_NEAREST)
        cv2.imshow(f"{filter_type.capitalize()} Filter", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def webcam_emotion_detection():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(gray, (48, 48))
        norm_face = face_resized / 255.0
        input_face = norm_face.reshape(1, 48, 48, 1)
        preds = model.predict(input_face)[0]
        emotion_idx = np.argmax(preds)
        label = class_labels[emotion_idx]
        confidence = round(preds[emotion_idx] * 100, 2)
        cv2.putText(frame, f"{label}: {confidence}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Webcam Emotion Detection (Press Q to exit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# --- GUI Setup ---
root = tk.Tk()
root.title("Emotion Recognition Classifier")
root.geometry("460x700")
root.configure(bg="#f0f0f0")

tk.Label(root, text="🤖 Facial Emotion Recognition", font=("Helvetica", 16, "bold"), bg="#f0f0f0", fg="#333").pack(pady=15)

tk.Button(root, text="📤 Upload Image & Predict", command=predict_emotion, bg="#4CAF50", fg="white", font=("Arial", 12), width=35).pack(pady=10)

tk.Label(root, text="🎨 Image Filters", font=("Arial", 12, "bold"), bg="#f0f0f0").pack(pady=(20, 5))

tk.Button(root, text="🌀 Apply Blur", command=lambda: apply_filter('blur'), width=30).pack(pady=2)
tk.Button(root, text="✴️ Apply Sharpen", command=lambda: apply_filter('sharpen'), width=30).pack(pady=2)
tk.Button(root, text="📐 Edge Detection", command=lambda: apply_filter('edge'), width=30).pack(pady=2)

tk.Label(root, text="🖼️ Uploaded Image Preview", font=("Arial", 11, "italic"), bg="#f0f0f0").pack(pady=(25, 5))
image_label = tk.Label(root, bg="#dcdcdc", width=150, height=150)
image_label.pack(pady=5)

tk.Label(root, text="📊 Top 3 Predicted Emotions", font=("Arial", 11, "italic"), bg="#f0f0f0").pack(pady=(20, 5))
result_text = tk.StringVar()
tk.Label(root, textvariable=result_text, font=("Arial", 12), bg="#f0f0f0", fg="#003366").pack()

tk.Button(root, text="🎥 Start Webcam Emotion Detection", command=webcam_emotion_detection, width=35, bg="#2196F3", fg="white").pack(pady=15)

tk.Label(root, text="© Ahsan Khizar - DIP Project", font=("Arial", 9), bg="#f0f0f0", fg="gray").pack(side="bottom", pady=10)

root.mainloop()
