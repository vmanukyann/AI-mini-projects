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
        self.canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)
        self.drawing = False
        self.last_pos = None
        
        # Drawing settings
        self.brush_size = 12
        self.brush_opacity = 0.8
        
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
    
    def draw_line(self, start_pos, end_pos):
        """Draw a smooth line between two points"""
        if start_pos is None:
            return
        
        x1, y1 = start_pos
        x2, y2 = end_pos
        
        # Calculate distance and steps for smooth line
        distance = max(abs(x2 - x1), abs(y2 - y1))
        if distance == 0:
            self.draw_brush(end_pos)
            return
        
        # Interpolate points along the line
        steps = max(1, int(distance))
        for i in range(steps + 1):
            t = i / steps if steps > 0 else 0
            x = int(x1 + (x2 - x1) * t)
            y = int(y1 + (y2 - y1) * t)
            self.draw_brush((x, y))
    
    def draw_brush(self, pos):
        """Draw a brush stroke at the given position with anti-aliasing"""
        x, y = pos
        if not (0 <= x < CANVAS_SIZE and 0 <= y < CANVAS_SIZE):
            return
        
        # Create a soft, anti-aliased brush
        for i in range(-self.brush_size, self.brush_size + 1):
            for j in range(-self.brush_size, self.brush_size + 1):
                px, py = x + i, y + j
                if 0 <= px < CANVAS_SIZE and 0 <= py < CANVAS_SIZE:
                    # Calculate distance from center
                    dist = np.sqrt(i*i + j*j)
                    
                    if dist <= self.brush_size:
                        # Soft falloff for anti-aliasing
                        if dist <= self.brush_size * 0.7:
                            intensity = self.brush_opacity
                        else:
                            # Smooth falloff at edges
                            falloff = 1.0 - (dist - self.brush_size * 0.7) / (self.brush_size * 0.3)
                            intensity = self.brush_opacity * max(0, falloff)
                        
                        # Blend with existing pixels
                        current_value = self.canvas[py, px]
                        new_value = min(1.0, current_value + intensity)
                        self.canvas[py, px] = new_value
    
    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas.fill(0)
        self.prediction = None
        self.confidence = 0
        self.last_pos = None
    
    def predict_digit(self):
        """Use the neural network to predict the drawn digit"""
        # Convert canvas to uint8 for PIL
        canvas_uint8 = (self.canvas * 255).astype(np.uint8)
        
        # Resize canvas to 28x28 for the model
        canvas_image = Image.fromarray(canvas_uint8)
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
        
        # Create canvas surface with smooth rendering
        canvas_surface = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
        canvas_surface.fill((30, 30, 30))  # Dark gray background
        
        # Draw the canvas with anti-aliasing
        for y in range(CANVAS_SIZE):
            for x in range(CANVAS_SIZE):
                intensity = self.canvas[y, x]
                if intensity > 0:
                    # White drawing on dark background
                    color_value = int(30 + intensity * 225)  # Blend from dark gray to white
                    color = (color_value, color_value, color_value)
                    canvas_surface.set_at((x, y), color)
        
        # Scale and center the canvas
        canvas_rect = pygame.Rect((WINDOW_SIZE - CANVAS_SIZE) // 2, 50, CANVAS_SIZE, CANVAS_SIZE)
        self.screen.blit(canvas_surface, canvas_rect)
        
        # Draw canvas border with rounded corners effect
        pygame.draw.rect(self.screen, BLACK, canvas_rect, 3)
        
        # Draw brush size indicator
        mouse_pos = pygame.mouse.get_pos()
        if canvas_rect.collidepoint(mouse_pos) and self.drawing:
            # Show brush preview
            brush_preview_surface = pygame.Surface((self.brush_size * 2, self.brush_size * 2), pygame.SRCALPHA)
            pygame.draw.circle(brush_preview_surface, (255, 255, 255, 100), 
                             (self.brush_size, self.brush_size), self.brush_size)
            preview_rect = brush_preview_surface.get_rect(center=mouse_pos)
            self.screen.blit(brush_preview_surface, preview_rect)
        
        # Draw buttons with better styling
        submit_rect = pygame.Rect(50, WINDOW_SIZE - 40, 120, 40)
        clear_rect = pygame.Rect(200, WINDOW_SIZE - 40, 120, 40)
        quit_rect = pygame.Rect(350, WINDOW_SIZE - 40, 120, 40)
        brush_up_rect = pygame.Rect(500, WINDOW_SIZE - 40, 40, 18)
        brush_down_rect = pygame.Rect(500, WINDOW_SIZE - 20, 40, 18)
        
        # Button colors with hover effects
        mouse_pos = pygame.mouse.get_pos()
        submit_color = (0, 120, 255) if submit_rect.collidepoint(mouse_pos) else BLUE
        clear_color = (150, 150, 150) if clear_rect.collidepoint(mouse_pos) else GRAY
        quit_color = (255, 50, 50) if quit_rect.collidepoint(mouse_pos) else (200, 0, 0)
        
        pygame.draw.rect(self.screen, submit_color, submit_rect, border_radius=5)
        pygame.draw.rect(self.screen, clear_color, clear_rect, border_radius=5)
        pygame.draw.rect(self.screen, quit_color, quit_rect, border_radius=5)
        pygame.draw.rect(self.screen, (100, 100, 255), brush_up_rect, border_radius=3)
        pygame.draw.rect(self.screen, (100, 100, 255), brush_down_rect, border_radius=3)
        
        # Button text
        submit_text = self.font.render("Submit", True, WHITE)
        clear_text = self.font.render("Clear", True, WHITE)
        quit_text = self.font.render("Quit", True, WHITE)
        plus_text = self.font.render("+", True, WHITE)
        minus_text = self.font.render("-", True, WHITE)
        
        self.screen.blit(submit_text, (submit_rect.x + 25, submit_rect.y + 10))
        self.screen.blit(clear_text, (clear_rect.x + 30, clear_rect.y + 10))
        self.screen.blit(quit_text, (quit_rect.x + 35, quit_rect.y + 10))
        self.screen.blit(plus_text, (brush_up_rect.x + 12, brush_up_rect.y - 2))
        self.screen.blit(minus_text, (brush_down_rect.x + 14, brush_down_rect.y - 2))
        
        # Draw title and instructions
        title_text = self.font.render("Draw a digit (0-9) and click Submit", True, BLACK)
        title_rect = title_text.get_rect(center=(WINDOW_SIZE // 2, 25))
        self.screen.blit(title_text, title_rect)
        
        # Brush size indicator
        brush_text = self.font.render(f"Brush: {self.brush_size}", True, BLACK)
        self.screen.blit(brush_text, (450, WINDOW_SIZE - 60))
        
        # Draw prediction result with better styling
        if self.prediction is not None:
            # Background for prediction
            pred_bg_rect = pygame.Rect(WINDOW_SIZE // 2 - 100, CANVAS_SIZE + 80, 200, 80)
            pygame.draw.rect(self.screen, (240, 255, 240), pred_bg_rect, border_radius=10)
            pygame.draw.rect(self.screen, GREEN, pred_bg_rect, 3, border_radius=10)
            
            pred_text = self.large_font.render(f"Prediction: {self.prediction}", True, GREEN)
            conf_text = self.font.render(f"Confidence: {self.confidence:.1f}%", True, GREEN)
            
            pred_rect = pred_text.get_rect(center=(WINDOW_SIZE // 2, CANVAS_SIZE + 100))
            conf_rect = conf_text.get_rect(center=(WINDOW_SIZE // 2, CANVAS_SIZE + 130))
            
            self.screen.blit(pred_text, pred_rect)
            self.screen.blit(conf_text, conf_rect)
        
        return submit_rect, clear_rect, quit_rect, canvas_rect, brush_up_rect, brush_down_rect
    
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
                        submit_rect, clear_rect, quit_rect, canvas_rect, brush_up_rect, brush_down_rect = self.draw_ui()
                        
                        if submit_rect.collidepoint(mouse_pos):
                            print("Predicting...")
                            self.predict_digit()
                            print(f"Prediction: {self.prediction}, Confidence: {self.confidence:.1f}%")
                        
                        elif clear_rect.collidepoint(mouse_pos):
                            print("Canvas cleared")
                            self.clear_canvas()
                        
                        elif quit_rect.collidepoint(mouse_pos):
                            running = False
                        
                        elif brush_up_rect.collidepoint(mouse_pos):
                            self.brush_size = min(20, self.brush_size + 2)
                        
                        elif brush_down_rect.collidepoint(mouse_pos):
                            self.brush_size = max(4, self.brush_size - 2)
                        
                        elif canvas_rect.collidepoint(mouse_pos):
                            self.drawing = True
                            # Convert screen coordinates to canvas coordinates
                            canvas_x = mouse_pos[0] - canvas_rect.x
                            canvas_y = mouse_pos[1] - canvas_rect.y
                            self.last_pos = (canvas_x, canvas_y)
                            self.draw_brush((canvas_x, canvas_y))
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.drawing = False
                        self.last_pos = None
                
                elif event.type == pygame.MOUSEMOTION:
                    if self.drawing:
                        mouse_pos = pygame.mouse.get_pos()
                        submit_rect, clear_rect, quit_rect, canvas_rect, brush_up_rect, brush_down_rect = self.draw_ui()
                        
                        if canvas_rect.collidepoint(mouse_pos):
                            # Convert screen coordinates to canvas coordinates
                            canvas_x = mouse_pos[0] - canvas_rect.x
                            canvas_y = mouse_pos[1] - canvas_rect.y
                            current_pos = (canvas_x, canvas_y)
                            
                            # Draw smooth line from last position
                            self.draw_line(self.last_pos, current_pos)
                            self.last_pos = current_pos
            
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