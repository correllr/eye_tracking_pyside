import toga
import numpy as np
import cv2
import tempfile
from toga import Window, Box, ImageView, Label
from toga.style import Pack

class TestApp(toga.App):
    def startup(self):
        self.main_window = Window("Test ImageView Click", size=(400, 300))
        
        # Create a dummy image: white background with "Test Image" text.
        dummy_img = np.full((300, 400, 3), 255, dtype=np.uint8)
        cv2.putText(dummy_img, "Test Image", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        # Save the dummy image to a temporary file.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            cv2.imwrite(f.name, dummy_img)
            image_path = f.name
        
        # Create an ImageView and load the image.
        self.image_view = ImageView(style=Pack(width=400, height=300, padding=0))
        self.image_view.image = toga.Image(image_path)
        
        # Create a container Box that will cover the ImageView.
        self.container = Box(style=Pack(width=400, height=300, padding=0))
        self.container.add(self.image_view)
        
        # Bind mouse events to the container.
        self.container.on_press = self.handle_press
        self.container.on_release = self.handle_release
        
        self.label = Label("Click on the image", style=Pack(padding=5))
        
        # Build the layout.
        main_box = Box(style=Pack(direction="column", padding=10))
        main_box.add(self.label)
        main_box.add(self.container)
        
        self.main_window.content = main_box
        self.main_window.show()

    def handle_press(self, widget, x, y):
        print(f"Pressed at: ({x}, {y})")

    def handle_release(self, widget, x, y):
        print(f"Released at: ({x}, {y})")

def main():
    return TestApp("Test App", "org.example.test")

if __name__ == '__main__':
    main().main_loop()