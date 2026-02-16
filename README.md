# Pong Deep Q-Learning (DQN)

This project implements a Reinforcement Learning agent that learns to play Pong from raw pixels using Deep Q-Learning (DQN). It features a custom Pong engine, a self-play training loop, and a real-time web dashboard.

## Installation

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Train the AI (`train.py`)

To start training the agent against itself:

```bash
python3 train.py
```

-   **Game Window**: A Pygame window will appear showing the training process (self-play).
-   **Dashboard**: Open your web browser and navigate to `http://localhost:5000` to view real-time metrics:
    -   **Score**: The score of the current episode.
    -   **Epsilon**: The current exploration rate.
    -   **Loss**: The training loss of the neural network.

The model is saved automatically as `pong_dqn.pth` every 10 episodes.

### 2. Play vs. AI (`play.py`)

Once a model is trained (or if you want to play against a random/untrained agent):

```bash
python3 play.py
```

-   **Controls**:
    -   **UP Arrow** / **W**: Move Left Paddle Up
    -   **DOWN Arrow** / **S**: Move Left Paddle Down
    -   **ESC**: Quit the game

## Project Structure

-   `pong_engine.py`: The custom Pong game environment.
-   `dqn_agent.py`: The Deep Q-Network implementation (CNN + Replay Buffer).
-   `train.py`: Main training loop with dashboard integration.
-   `play.py`: Script to play against the trained model.
-   `dashboard.py`: Flask application for the web dashboard.
-   `templates/dashboard.html`: Frontend for the dashboard.
