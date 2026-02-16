import pygame
import torch
import numpy as np
import collections
import cv2
import os
import sys

# Add project root to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pong_engine import PongGame
from dqn_agent import DQNAgent

# Hyperparameters
FPS = 60

def play():
    # Initialize environment
    # Ensure render_mode=True for visual feedback
    env = PongGame(render_mode=True)

    # Initialize Agent
    agent = DQNAgent(input_shape=(4, 84, 84), num_actions=3)

    # Load Model
    model_path = "pong_dqn.pth"
    if os.path.exists(model_path):
        agent.load(model_path)
        print("Model loaded successfully.")
    else:
        print("No trained model found. Playing against random agent.")

    clock = pygame.time.Clock()
    running = True

    while running:
        frame = env.reset()
        if frame is None:
            break

        # Stack 4 frames
        state_buffer = collections.deque([frame] * 4, maxlen=4)

        def get_state(buffer):
            return np.array(buffer, dtype=np.uint8)

        current_state = get_state(state_buffer)

        done = False
        while not done:
            # Handle Human Input (Left Paddle)
            action_left = 0
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                action_left = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                action_left = 2

            if keys[pygame.K_ESCAPE]:
                running = False
                break

            # Handle AI Input (Right Paddle)
            # Flip state for AI perspective
            state_flipped = np.flip(current_state, axis=2).copy()

            # Epsilon = 0 (Greedy)
            # Make sure to handle torch.device correctly inside agent
            # If no model loaded, act random?
            # act() handles epsilon. If epsilon=0, it uses network.
            # If network is random (init), it acts random but consistent.
            action_right = agent.act(state_flipped, epsilon=0.0)

            # Step
            next_frame, reward_left, reward_right, done = env.step(action_left, action_right)

            if next_frame is None: # Window closed
                running = False
                break

            # Update buffer
            state_buffer.append(next_frame)
            current_state = get_state(state_buffer)

            # Cap FPS
            clock.tick(FPS)

        if not running:
            break

        print(f"Game Over! Score: Human {env.left_score} - AI {env.right_score}")
        pygame.time.wait(2000) # Wait 2 seconds before restart

    env.close()

if __name__ == "__main__":
    play()
