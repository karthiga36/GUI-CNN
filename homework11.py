import sys
import tensorflow as tf
from tensorflow.keras import layers, models
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PIL import Image
import numpy as np

#Fashion MNIST Model training
def train_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


    x_train, x_test = x_train / 255.0, x_test / 255.0


    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    model.fit(x_train, y_train, epochs=10)


    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    #TFLite model
    with open('fashion_mnist_model.tflite', 'wb') as f:
        f.write(tflite_model)

    return model

interpreter = tf.lite.Interpreter(model_path="fashion_mnist_model.tflite")
interpreter.allocate_tensors()

# predict the image class using TFLite model
def predict_image(image_path):
    # Load and preprocess the image
    img = Image.open(image_path).convert('L').resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28)
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = np.argmax(output_data)
    return predicted_class

# Create the GUI using PyQt
class ImageClassifier(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Fashion MNIST Image Classifier')
        self.setGeometry(100, 100, 400, 400)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.image_label = QLabel('Drag & drop your image here', self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)
        self.result_label = QLabel('Predicted Class: None', self)
        self.layout.addWidget(self.result_label)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
      
        image_path = event.mimeData().urls()[0].toLocalFile()
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))
        predicted_class = predict_image(image_path)
        self.result_label.setText(f'Predicted Class: {predicted_class}')

def main():
    # Train model and save it as TFLite
    train_model()
    app = QApplication(sys.argv)
    window = ImageClassifier()
    window.show()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
