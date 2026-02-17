import threading
import time
import numpy as np
import torch
import collections
import cv2
import os
import sys

# Add project root to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pong_engine import PongGame
from dqn_agent import DQNAgent
import dashboard

# Hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY = 100000
TARGET_UPDATE = 1000
MEMORY_SIZE = 10000
LEARNING_RATE = 1e-4
MAX_EPISODES = 5000

def train():
    # Start dashboard
    def start_dashboard():
        dashboard.run_dashboard(port=50000)

    t = threading.Thread(target=start_dashboard)
    t.daemon = True
    t.start()
    print("Dashboard running at http://localhost:5000")

    # Initialize environment
    # Use render_mode=True to visualize learning.
    env = PongGame(render_mode=True)

    agent = DQNAgent(input_shape=(4, 84, 84), num_actions=3, learning_rate=LEARNING_RATE, buffer_size=MEMORY_SIZE)

    epsilon = EPS_START
    total_steps = 0

    for episode in range(MAX_EPISODES):
        frame = env.reset()

        # Stack 4 frames
        state_buffer = collections.deque([frame] * 4, maxlen=4)

        def get_state(buffer):
            return np.array(buffer, dtype=np.uint8)

        current_state = get_state(state_buffer)

        total_reward_left = 0
        total_reward_right = 0
        loss_val = 0
        steps_in_episode = 0

        done = False
        while not done:
            # Select action for Left Paddle (Agent)
            # Epsilon decay
            epsilon = max(EPS_END, EPS_START - (total_steps / EPS_DECAY))

            action_left = agent.act(current_state, epsilon)

            # Select action for Right Paddle (Agent playing as Right)
            # Flip state horizontally (axis 2 is width)
            state_flipped = np.flip(current_state, axis=2).copy()
            action_right = agent.act(state_flipped, epsilon)

            # Step
            next_frame, reward_left, reward_right, done = env.step(action_left, action_right)

            if done and next_frame is None: # Quit event
                break

            # Update buffer
            state_buffer.append(next_frame)
            next_state = get_state(state_buffer)

            # Store Left Experience
            agent.memory.push(current_state, action_left, reward_left, next_state, done)

            # Store Right Experience (flipped)
            next_state_flipped = np.flip(next_state, axis=2).copy()
            # Note: Actions 1 (Up) and 2 (Down) are same for both paddles relative to their side.
            agent.memory.push(state_flipped, action_right, reward_right, next_state_flipped, done)

            # Train
            loss = agent.learn(BATCH_SIZE)
            if loss is not None:
                loss_val += loss

            # Update state
            current_state = next_state

            total_reward_left += reward_left
            total_reward_right += reward_right
            total_steps += 1
            steps_in_episode += 1

            # Update Target Network
            if total_steps % TARGET_UPDATE == 0:
                agent.update_target_network()

        if done and next_frame is None:
            break

        # End of Episode
        avg_loss = loss_val / steps_in_episode if steps_in_episode > 0 else 0
        # Use simple score difference or just left score
        score = env.left_score

        print(f"Episode {episode}: Score {score}, Eps: {epsilon:.4f}, Loss: {avg_loss:.4f}")

        dashboard.update_metrics(episode, score, epsilon, avg_loss)

        # Save model every 10 episodes
        if episode % 10 == 0:
            agent.save("pong_dqn.pth")

    env.close()
    print("Training Complete")

if __name__ == "__main__":
    train()
