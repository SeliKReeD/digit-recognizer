import sys
import io
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt, QBuffer
from PIL import Image
from keras.models import load_model
import numpy as np

# Array of progress bars which shows probability of being any of 10 number.
probabilities = []


# Paint area class.
class Canvas(QtWidgets.QLabel):

    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: white;")
        # Create pixmap for drawing digits.
        pixmap = QtGui.QPixmap(280, 280)
        self.setPixmap(pixmap)
        self.last_x, self.last_y = None, None
        # Make pen color white.
        self.pen_color = QtGui.QColor('#ffffff')
        self.model = load_model('data/snapshots/20-02-2021-21-02-30-SNAPSHOT-l-0.035-a-0.99.h5')

    def mouseMoveEvent(self, e):
        # First drawing.
        if self.last_x is None:
            self.last_x = e.x()
            self.last_y = e.y()
            # Ignore the first time.
            return

        # Configure painter for drawing.
        painter = QtGui.QPainter(self.pixmap())
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.HighQualityAntialiasing, True)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        p = painter.pen()
        p.setWidth(15)
        p.setColor(self.pen_color)
        p.setCapStyle(Qt.RoundCap)
        p.setJoinStyle(Qt.RoundJoin)
        painter.setPen(p)
        # Draw line between previous event point and current one.
        painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        painter.end()
        self.update()
        # Convert pixmap image to PIL image for doing ML stuff.
        img = self.pixmap().toImage()
        buffer = QBuffer()
        buffer.open(QBuffer.ReadWrite)
        img.save(buffer, "PNG")
        pil_im = Image.open(io.BytesIO(buffer.data()))
        # Resize image to fit input shape of pretrained model [28, 28].
        img = pil_im.convert("L").resize(size=(28, 28), resample=Image.NEAREST)
        # Convert image to numpy array for feeding it to neural network model.
        im2arr = np.array(img)
        # Normalize pixel data by dividing by 255. It will give us values between 0(black) and 1(white).
        im2arr = np.divide(im2arr, 255)
        # Reshape array. Adding one dimension fo color - grayscale.
        im2arr = im2arr.reshape((1, 28, 28, 1))
        # Make prediction.
        y_pred2 = self.model.predict(im2arr)
        # Convert results from ndarray to array.
        prediction_array = y_pred2.ravel()
        # Display prediction results on probabilities progress bars.
        for i in range(10):
            probabilities[i].setValue(prediction_array[i] * 100)
        # Update the origin point for next draw event.
        self.last_x = e.x()
        self.last_y = e.y()

    # Reset origin points.
    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None

    # Reset origin points and fill pixmap with black.
    def clear(self):
        self.pixmap().fill(Qt.black)
        self.update()
        self.last_x, self.last_y = None, None
        for i in range(10):
            probabilities[i].setValue(0)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.canvas = Canvas()
        w = QtWidgets.QWidget()
        # Main layout for all elements. Will contain canvas_layout and sliders_layout.
        main_layout = QtWidgets.QHBoxLayout()
        w.setLayout(main_layout)
        # Layout for containing progress bars with digit probabilities.
        sliders_layout = QtWidgets.QVBoxLayout()
        # Place in progress bars and labels attached to them.
        for n in range(10):
            # Create label.
            label = QtWidgets.QLabel()
            # Set label text to number from 0 to 9.
            label.setText(str(n))
            # Create horizontal layout for label and progress bar.
            slider_layout = QtWidgets.QHBoxLayout()
            # Add label to this layout.
            slider_layout.addWidget(label)
            # Create progress bar. Make min/max values 0/100, set size of progress bar.
            progress_bar = QtWidgets.QProgressBar()
            progress_bar.setMinimum(0)
            progress_bar.setMaximum(100)
            progress_bar.setFixedSize(200, 20)
            # Add progress bar to layout.
            slider_layout.addWidget(progress_bar)
            # Add local layout to sliders_layout for all progress bars.
            sliders_layout.addLayout(slider_layout)
            # Add progress bar to array of all progress bars.
            probabilities.append(progress_bar)
        # Make separate layout for canvas and Clear button.
        canvas_layout = QtWidgets.QVBoxLayout()
        # Add drawing area to this layout.
        canvas_layout.addWidget(self.canvas)
        # Create Clear button.
        clear_button = QtWidgets.QPushButton()
        clear_button.setText("Clear")
        # Attach to pressed event clear() method.
        clear_button.pressed.connect(lambda: self.canvas.clear())
        # Add button to layout.
        canvas_layout.addWidget(clear_button)
        # Add all layouts to main_layout. Final UI composing.
        main_layout.addLayout(canvas_layout)
        main_layout.addLayout(sliders_layout)
        self.setCentralWidget(w)


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()
