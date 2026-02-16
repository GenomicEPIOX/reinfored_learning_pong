import pygame
import random
import numpy as np
import cv2
import os

# Constants
WIDTH, HEIGHT = 640, 480
PADDLE_WIDTH, PADDLE_HEIGHT = 15, 60
BALL_SIZE = 10
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class PongGame:
    def __init__(self, window_width=WIDTH, window_height=HEIGHT, max_score=5, render_mode=False):
        self.window_width = window_width
        self.window_height = window_height
        self.max_score = max_score
        self.render_mode = render_mode

        # Check if running in a headless environment
        if os.environ.get('SDL_VIDEODRIVER') == 'dummy' or not os.environ.get('DISPLAY'):
            os.environ['SDL_VIDEODRIVER'] = 'dummy'

        pygame.init()
        # Initialize screen for rendering
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Pong DQN")
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        # Paddle positions
        self.left_paddle_y = self.window_height // 2 - PADDLE_HEIGHT // 2
        self.right_paddle_y = self.window_height // 2 - PADDLE_HEIGHT // 2

        # Ball state
        self.ball_x = self.window_width // 2 - BALL_SIZE // 2
        self.ball_y = self.window_height // 2 - BALL_SIZE // 2

        # Randomize ball direction
        self.ball_dx = 5 * random.choice([1, -1])
        self.ball_dy = 5 * random.choice([1, -1])

        self.left_score = 0
        self.right_score = 0

        return self.get_frame()

    def _reset_ball(self):
        self.ball_x = self.window_width // 2 - BALL_SIZE // 2
        self.ball_y = self.window_height // 2 - BALL_SIZE // 2
        self.ball_dx *= -1
        self.ball_dy = 5 * random.choice([1, -1])

    def step(self, left_action, right_action):
        # Actions: 0 = Stay, 1 = Up, 2 = Down
        paddle_speed = 6

        # Left Paddle Movement
        if left_action == 1: # UP
            self.left_paddle_y = max(0, self.left_paddle_y - paddle_speed)
        elif left_action == 2: # DOWN
            self.left_paddle_y = min(self.window_height - PADDLE_HEIGHT, self.left_paddle_y + paddle_speed)

        # Right Paddle Movement
        if right_action == 1: # UP
            self.right_paddle_y = max(0, self.right_paddle_y - paddle_speed)
        elif right_action == 2: # DOWN
            self.right_paddle_y = min(self.window_height - PADDLE_HEIGHT, self.right_paddle_y + paddle_speed)

        # Ball Movement
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        # Wall Collisions (Top/Bottom)
        if self.ball_y <= 0 or self.ball_y >= self.window_height - BALL_SIZE:
            self.ball_dy *= -1

        # Paddle Collisions & Rewards
        reward_left = 0
        reward_right = 0

        # Left Paddle
        left_paddle_rect = pygame.Rect(10, self.left_paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT)
        ball_rect = pygame.Rect(self.ball_x, self.ball_y, BALL_SIZE, BALL_SIZE)

        if ball_rect.colliderect(left_paddle_rect) and self.ball_dx < 0:
            self.ball_dx *= -1.1 # Speed up slightly
            offset = (self.ball_y + BALL_SIZE/2) - (self.left_paddle_y + PADDLE_HEIGHT/2)
            self.ball_dy += offset * 0.1
            reward_left += 0.1 # Reward for hitting ball

        # Right Paddle
        right_paddle_rect = pygame.Rect(self.window_width - 25, self.right_paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT)

        if ball_rect.colliderect(right_paddle_rect) and self.ball_dx > 0:
            self.ball_dx *= -1.1
            offset = (self.ball_y + BALL_SIZE/2) - (self.right_paddle_y + PADDLE_HEIGHT/2)
            self.ball_dy += offset * 0.1
            reward_right += 0.1 # Reward for hitting ball

        # Cap ball speed
        self.ball_dx = max(min(self.ball_dx, 15), -15)
        self.ball_dy = max(min(self.ball_dy, 10), -10)

        # Scoring
        done = False

        if self.ball_x < 0:
            self.right_score += 1
            reward_left = -1
            reward_right = 1
            self._reset_ball()
        elif self.ball_x > self.window_width:
            self.left_score += 1
            reward_left = 1
            reward_right = -1
            self._reset_ball()

        if self.left_score >= self.max_score or self.right_score >= self.max_score:
            done = True

        # Handle Pygame events (keep window responsive)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                pygame.quit()
                return None, 0, 0, True

        self.render()

        if self.render_mode:
             self.clock.tick(FPS)

        next_frame = self.get_frame()
        return next_frame, reward_left, reward_right, done

    def render(self):
        try:
            self.screen.fill(BLACK)

            # Draw paddles
            pygame.draw.rect(self.screen, WHITE, (10, self.left_paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT))
            pygame.draw.rect(self.screen, WHITE, (self.window_width - 25, self.right_paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT))

            # Draw ball
            pygame.draw.rect(self.screen, WHITE, (self.ball_x, self.ball_y, BALL_SIZE, BALL_SIZE))

            # Draw Score
            font = pygame.font.Font(None, 74)
            text = font.render(str(self.left_score), 1, WHITE)
            self.screen.blit(text, (self.window_width // 4, 10))
            text = font.render(str(self.right_score), 1, WHITE)
            self.screen.blit(text, (3 * self.window_width // 4, 10))

            pygame.display.flip()
        except pygame.error:
            pass

    def get_frame(self):
        try:
            # Get raw pixels from the screen surface
            view = pygame.surfarray.array3d(self.screen)
            # Transpose to (Height, Width, Channels) because surfarray is (Width, Height, Channels)
            view = view.transpose([1, 0, 2])

            # Convert to grayscale
            img = cv2.cvtColor(view, cv2.COLOR_RGB2GRAY)

            # Resize to 84x84 (Standard DQN input)
            img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)

            return img.astype(np.uint8)
        except pygame.error:
            # If display surface is gone
            return np.zeros((84, 84), dtype=np.uint8)

    def close(self):
        pygame.quit()
