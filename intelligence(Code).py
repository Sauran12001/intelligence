import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
import numpy as np
import tkinter as tk
from tkinter import Toplevel, StringVar, messagebox
from PIL import Image, ImageDraw

# بارگذاری و نرمال‌سازی داده‌ها
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# ساخت مدل شبکه عصبی
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

# کامپایل و آموزش مدل
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

class DigitRecognizer:
    def __init__(self, master):
        self.master = master
        self.master.title("Digit Recognizer")
        self.canvas = tk.Canvas(self.master, width=200, height=200, bg='white')
        self.canvas.pack()
        self.canvas.bind('<B1-Motion>', self.draw)
        self.button_frame = tk.Frame(self.master)
        self.button_frame.pack()
        self.predict_button = tk.Button(self.button_frame, text="Predict", command=self.predict)
        self.predict_button.pack(side=tk.LEFT)
        self.clear_button = tk.Button(self.button_frame, text="Clear", command=self.clear)
        self.clear_button.pack(side=tk.LEFT)
        self.image = Image.new('L', (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.last_prediction = None
        self.last_accuracy = None
        self.total_neurons = sum([layer.units for layer in model.layers if isinstance(layer, Dense)])
        self.active_neurons = self.total_neurons

    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-6, y-6, x+6, y+6, fill='black')
        self.draw.ellipse([x-6, y-6, x+6, y+6], fill='black')

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new('L', (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)
        if self.last_accuracy is not None:
            self.display_accuracy_reduction()
        self.last_prediction = None
        self.last_accuracy = None

    def predict(self):
        self.image = self.image.resize((28, 28))
        img_array = np.array(self.image)
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)
        accuracy = np.max(prediction)
        self.display_prediction(predicted_digit, accuracy)
        if accuracy < 0.75:
            self.retrain_model()
        self.last_prediction = predicted_digit
        self.last_accuracy = accuracy

    def display_prediction(self, digit, accuracy):
        prediction_window = Toplevel(self.master)
        prediction_window.title("Prediction")
        label = tk.Label(prediction_window, text=f'Predicted Digit: {digit}\nAccuracy: {accuracy:.2f}', font=("Helvetica", 48))
        label.pack(pady=20)

    def retrain_model(self):
        status_window = Toplevel(self.master)
        status_window.title("Neuron Status")
        status_var = StringVar()
        status_label = tk.Label(status_window, textvariable=status_var, font=("Helvetica", 16))
        status_label.pack(pady=20)
        deactivated_count = 0
        for layer in model.layers:
            if isinstance(layer, Dense):
                active_neurons = layer.units
                deactivated_neurons = np.random.choice([True, False], active_neurons)
                layer.trainable = deactivated_neurons.sum() / active_neurons < 0.5
                deactivated_count += active_neurons - deactivated_neurons.sum()
        self.active_neurons -= deactivated_count
        status_var.set(f'Total Neurons: {self.total_neurons}\nActive Neurons: {self.active_neurons}')
        model.fit(x_train, y_train, epochs=1)
        self.clear()

    def display_accuracy_reduction(self):
        accuracy_reduction = (1 - self.last_accuracy) * 100
        messagebox.showinfo("Accuracy Reduction", f"Accuracy Reduction: {accuracy_reduction:.2f}%")

# ایجاد رابط کاربری
root = tk.Tk()
app = DigitRecognizer(root)
root.mainloop()
