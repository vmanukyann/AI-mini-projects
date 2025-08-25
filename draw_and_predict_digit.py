import pygame
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import sys

# Initialize Pygame
pygame.init()

# Constants
WINDOW_SIZE = 600
CANVAS_SIZE = 280
BUTTON_HEIGHT = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
BLUE = (0, 100, 200)
GREEN = (0, 200, 0)

class DigitRecognitionApp:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + BUTTON_HEIGHT))
        pygame.display.set_caption("Draw a Digit - Neural Network Prediction")
        
        # Canvas for drawing (28x28 will be the final size for the model)
        self.canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
        self.drawing = False
        
        # Load or create a simple neural network model
        self.model = self.create_model()
        
        # Font for text
        self.font = pygame.font.Font(None, 36)
        self.large_font = pygame.font.Font(None, 72)
        
        # Prediction result
        self.prediction = None
        self.confidence = 0
        
    def create_model(self):
        """Create and train a simple neural network on MNIST data"""
        try:
            # Try to load MNIST dataset
            (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
            
            # Normalize the data
            x_train = x_train.astype('float32') / 255.0
            x_test = x_test.astype('float32') / 255.0
            
            # Create a simple neural network
            model = keras.Sequential([
                keras.layers.Flatten(input_shape=(28, 28)),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(10, activation='softmax')
            ])
            
            model.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
            
            print("Training neural network... This may take a moment.")
            # Train the model (reduced epochs for faster startup)
            model.fit(x_train, y_train, epochs=3, batch_size=128, verbose=1,
                     validation_data=(x_test, y_test))
            
            print("Model training complete!")
            return model
            
        except Exception as e:
            print(f"Error creating model: {e}")
            print("Creating a dummy model for demonstration...")
            # Create a dummy model if MNIST loading fails
            model = keras.Sequential([
                keras.layers.Flatten(input_shape=(28, 28)),
                keras.layers.Dense(10, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            return model
    
    def draw_on_canvas(self, pos):
        """Draw on the canvas at the given position"""
        x, y = pos
        if 0 <= x < CANVAS_SIZE and 0 <= y < CANVAS_SIZE:
            # Draw a thick brush
            brush_size = 15
            for i in range(-brush_size, brush_size + 1):
                for j in range(-brush_size, brush_size + 1):
                    if (i*i + j*j) <= brush_size*brush_size:
                        px, py = x + i, y + j
                        if 0 <= px < CANVAS_SIZE and 0 <= py < CANVAS_SIZE:
                            self.canvas[py, px] = min(255, self.canvas[py, px] + 50)
    
    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas.fill(0)
        self.prediction = None
        self.confidence = 0
    
    def predict_digit(self):
        """Use the neural network to predict the drawn digit"""
        # Resize canvas to 28x28 for the model
        canvas_image = Image.fromarray(self.canvas)
        canvas_resized = canvas_image.resize((28, 28), Image.Resampling.LANCZOS)
        canvas_array = np.array(canvas_resized, dtype=np.float32) / 255.0
        
        # Add batch dimension
        input_data = canvas_array.reshape(1, 28, 28)
        
        # Make prediction
        predictions = self.model.predict(input_data, verbose=0)
        self.prediction = np.argmax(predictions[0])
        self.confidence = np.max(predictions[0]) * 100
    
    def draw_ui(self):
        """Draw the user interface"""
        # Fill background
        self.screen.fill(WHITE)
        
        # Draw canvas area
        canvas_surface = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
        for y in range(CANVAS_SIZE):
            for x in range(CANVAS_SIZE):
                gray_value = 255 - self.canvas[y, x]  # Invert for display
                canvas_surface.set_at((x, y), (gray_value, gray_value, gray_value))
        
        # Scale and center the canvas
        canvas_rect = pygame.Rect((WINDOW_SIZE - CANVAS_SIZE) // 2, 50, CANVAS_SIZE, CANVAS_SIZE)
        self.screen.blit(canvas_surface, canvas_rect)
        
        # Draw canvas border
        pygame.draw.rect(self.screen, BLACK, canvas_rect, 3)
        
        # Draw buttons
        submit_rect = pygame.Rect(50, WINDOW_SIZE - 40, 120, 40)
        clear_rect = pygame.Rect(200, WINDOW_SIZE - 40, 120, 40)
        quit_rect = pygame.Rect(350, WINDOW_SIZE - 40, 120, 40)
        
        pygame.draw.rect(self.screen, BLUE, submit_rect)
        pygame.draw.rect(self.screen, GRAY, clear_rect)
        pygame.draw.rect(self.screen, (200, 0, 0), quit_rect)
        
        # Button text
        submit_text = self.font.render("Submit", True, WHITE)
        clear_text = self.font.render("Clear", True, WHITE)
        quit_text = self.font.render("Quit", True, WHITE)
        
        self.screen.blit(submit_text, (submit_rect.x + 25, submit_rect.y + 10))
        self.screen.blit(clear_text, (clear_rect.x + 30, clear_rect.y + 10))
        self.screen.blit(quit_text, (quit_rect.x + 35, quit_rect.y + 10))
        
        # Draw title
        title_text = self.font.render("Draw a digit (0-9) and click Submit", True, BLACK)
        title_rect = title_text.get_rect(center=(WINDOW_SIZE // 2, 25))
        self.screen.blit(title_text, title_rect)
        
        # Draw prediction result
        if self.prediction is not None:
            pred_text = self.large_font.render(f"Prediction: {self.prediction}", True, GREEN)
            conf_text = self.font.render(f"Confidence: {self.confidence:.1f}%", True, GREEN)
            
            pred_rect = pred_text.get_rect(center=(WINDOW_SIZE // 2, CANVAS_SIZE + 100))
            conf_rect = conf_text.get_rect(center=(WINDOW_SIZE // 2, CANVAS_SIZE + 140))
            
            self.screen.blit(pred_text, pred_rect)
            self.screen.blit(conf_text, conf_rect)
        
        return submit_rect, clear_rect, quit_rect, canvas_rect
    
    def run(self):
        """Main game loop"""
        clock = pygame.time.Clock()
        running = True
        
        print("Application started! Draw a digit and click Submit to see the prediction.")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        mouse_pos = pygame.mouse.get_pos()
                        submit_rect, clear_rect, quit_rect, canvas_rect = self.draw_ui()
                        
                        if submit_rect.collidepoint(mouse_pos):
                            print("Predicting...")
                            self.predict_digit()
                            print(f"Prediction: {self.prediction}, Confidence: {self.confidence:.1f}%")
                        
                        elif clear_rect.collidepoint(mouse_pos):
                            print("Canvas cleared")
                            self.clear_canvas()
                        
                        elif quit_rect.collidepoint(mouse_pos):
                            running = False
                        
                        elif canvas_rect.collidepoint(mouse_pos):
                            self.drawing = True
                            # Convert screen coordinates to canvas coordinates
                            canvas_x = mouse_pos[0] - canvas_rect.x
                            canvas_y = mouse_pos[1] - canvas_rect.y
                            self.draw_on_canvas((canvas_x, canvas_y))
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.drawing = False
                
                elif event.type == pygame.MOUSEMOTION:
                    if self.drawing:
                        mouse_pos = pygame.mouse.get_pos()
                        submit_rect, clear_rect, quit_rect, canvas_rect = self.draw_ui()
                        
                        if canvas_rect.collidepoint(mouse_pos):
                            # Convert screen coordinates to canvas coordinates
                            canvas_x = mouse_pos[0] - canvas_rect.x
                            canvas_y = mouse_pos[1] - canvas_rect.y
                            self.draw_on_canvas((canvas_x, canvas_y))
            
            # Draw everything
            self.draw_ui()
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    print("Starting Digit Recognition App...")
    print("Required packages: pygame, tensorflow, pillow, numpy")
    print("Make sure you have these installed: pip install pygame tensorflow pillow numpy")
    
    try:
        app = DigitRecognitionApp()
        app.run()
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install required packages with: pip install pygame tensorflow pillow numpy")
    except Exception as e:
        print(f"Error running application: {e}")