# IANNWTF_FinalProject

## Introduction
In this project we implemented a Deep-Q-Network and a Double Deep-Q-Network to solve the Lunar Lander Gym Environment. 

## Lunar Lander Environment

### Action Space
The Lunar Lander environment provides four discrete actions:
- 0: Do nothing
- 1: Fire left orientation engine
- 2: Fire main engine
- 3: Fire right orientation engine

### Observation Space
The state in the Lunar Lander environment is represented by an 8-dimensional vector, including:
- Coordinates of the lander in x and y
- Linear velocities in x and y
- Angle of the lander
- Angular velocity of the lander
- Boolean indicating whether each leg is in contact with the ground or not

### Rewards
After every step in the Lunar Lander environment, a reward is granted. The total reward for an episode is the sum of rewards for all steps within that episode.

For each step, the reward:
- Increases/decreases based on the proximity/distance of the lander to the landing pad.
- Increases/decreases based on the speed of the lander.
- Decreases as the lander becomes more tilted (angle not horizontal).
- Increases by 10 points for each leg in contact with the ground.
- Decreases by 0.03 points for each frame a side engine is firing.
- Decreases by 0.3 points for each frame the main engine is firing.

Additionally, an episode receives an additional reward of -100 or +100 points for crashing or landing safely respectively. An episode is considered solved if it scores at least 200 points.

## How to Run the Model
To run the model, follow these steps:

1. **Install Dependencies**: First, you need to install all dependencies. You can do this by running the following command in your terminal:
    ```bash
    pip install -r "requirements.txt"
    ```
   
2. **Run the Model**: After installing the dependencies, run the following command to try out our models:
    ```bash
    python3 test.py
    ```

3. **Choose Model**:
    - When prompted, you can choose between the following options:
        1. DQN
        2. DDQN

    Enter the corresponding number (1 or 2) to select the desired model.
