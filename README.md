# Reusable Rocket Simulation

## About this project

Driven by my admiration for SpaceX and a lifelong fascination with rocketry, I've embarked on an odyssey uniting virtual rocketry with cutting-edge innovation. Can the fusion of virtual rocket crafting and reinforcement learning decode rocket recycling? With dreams of my own rocket venture, I'm sculpting intricate digital rocket avatars, honing designs free from physical restraints. This journey extends to the heart of space exploration: recycling rockets for sustainability. By applying reinforcement learning, I aim to empower virtual rockets to autonomously return to Earth, reshaping rocket refurbishment

![](./gallery/config.jpg)



For the landing task, I followed the basic parameters of the Starship SN10 belly flop maneuver. The initial speed is set to -50m/s. The rocket orientation is set to 90 degrees (horizontally). The landing burn height is set to 500 meters above the ground. 

![](./gallery/timelapse.jpg)


The reward functions are quite straightforward.

For the hovering tasks: the step-reward is given based on two rules: 1) The distance between the rocket and the predefined target point - the closer they are, the larger reward will be assigned. 2) The angle of the rocket body (the rocket should stay as upright as possible)

For the landing task: we look at the Speed and angle at the moment of contact with the ground - when the touching-speed are smaller than a safe threshold and the angle is close to 0 degrees (upright), we see it as a successful landing and a big reward will be assigned. The rest of the rules are the same as the hovering task.


I implement the above environment and train a policy-based agent (actor-critic) to solve this problem. The episode reward finally converges very well after over 20000 training episodes.

## Fully trained agent (task: hovering) 


![h_20k](https://github.com/Anushka091922/Reusable-rockets-simulation/assets/114327511/fc4fb698-16e2-48e3-89d6-71335e04ef18)




  ## Landing Rewards


![landing_rewards_00011201](https://github.com/Anushka091922/Reusable-rockets-simulation/assets/114327511/309c74ad-a4ee-4e2a-b274-59be2585b397)


##  Fully trained agent (task: landing) |              |
 :-![l_11k](https://github.com/Anushka091922/Reusable-rockets-simulation/assets/114327511/6f4a0255-861b-4cd0-bd24-12d1fbdc6327)
---------------------------------: | :------------------------------------: |

##  Reward over number of episodes 

![hovering_rewards_00022301](https://github.com/Anushka091922/Reusable-rockets-simulation/assets/114327511/24201462-85d0-403a-bc01-cd543378036c)

Despite the simple setting of the environment and the reward, the agent has learned the belly flop maneuver nicely. The following animation shows a comparison between the real SN10 and a fake one learned from reinforcement learning.




![belly_flop](https://github.com/Anushka091922/Reusable-rockets-simulation/assets/114327511/49a9c793-169b-4f4c-a3f9-0980468b9759)


## Developing the "Rocket-recycling with Reinforcement Learning" project involves several steps that can be outlined as follows:

### Conceptualization and Research:

### Gain a deep understanding of reinforcement learning concepts
especially policy-based methods like Actor-Critic.
Research the physics and dynamics of rocket flight, landing maneuvers, and the principles of rocket recycling.
Environment Design and Implementation:

### Design the virtual rocket environment 
specifying the state space, action space, and the rules governing the rocket's behavior.
Implement the environment using a programming language (such as Python) and suitable libraries (e.g., Gym, PyTorch).
### Agent Architecture and Training:

Design the architecture of the reinforcement learning agent(Actor-Critic) that will learn to control the virtual rocket.
Set up the training loop, including episode initialization, action selection, reward computation, and backpropagation.
### Reward Engineering:

Devise reward functions that reflect the desired behavior for both hovering and landing tasks.
Experiment with different reward designs to encourage safe and effective rocket behavior.
### Training and Optimization:

Train the agent using various techniques, such as Proximal Policy Optimization (PPO) or other policy gradient methods.
Tweak hyperparameters, such as learning rates, discount factors, and exploration rates, to optimize training performance.
### Visualization and Analysis:

Implement visualization tools to monitor the agent's progress during training, including rewards and performance metrics.
Analyze the learning curves, convergence, and any signs of instability or overfitting.
### Model Saving and Loading:

Implement functionality to save the trained agent's model checkpoints for future use.
Develop a loading mechanism to reload trained models for testing and evaluation.
### Testing and Evaluation:

Evaluate the trained agent in various scenarios, such as different initial conditions and environments.
Compare the agent's behavior with real-world rocket maneuvers.
### Documentation:

Document the entire project, including the motivation, problem statement, methodology, implementation details, and results.
Create a comprehensive README file that guides users on how to set up, train, and test the agent.
### Fine-Tuning and Iteration:

Based on the evaluation results, fine-tune the agent's training parameters, environment settings, and reward functions.
Iterate on the project to enhance the agent's performance and address any limitations.
### Presentation and Sharing:

Create a demo video showcasing the agent's performance in both hovering and landing tasks.
Share the project, including the code, documentation, and demo video, on platforms like GitHub to contribute to the AI and space exploration communities.
### Further Exploration:

Explore more advanced reinforcement learning algorithms, such as Deep Deterministic Policy Gradients (DDPG) or Trust Region Policy Optimization (TRPO), to improve the agent's learning efficiency.

## courtesy:Chatgpt )

## configurations :
![config](https://github.com/Anushka091922/Reusable-rockets-simulation/assets/114327511/5a4b8eab-e79a-4b6d-ad47-ea952d40cc87)

## Timelapse:
![timelapse](https://github.com/Anushka091922/Reusable-rockets-simulation/assets/114327511/cf13dff2-16dd-4a3a-82f3-39eedfb674a1)



## Usage

To train an agent, see `./example_train.py`

To test an agent:

```python
import torch
from rocket import Rocket
from policy import ActorCritic
import os
import glob

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    task = 'hover'  # 'hover' or 'landing'
    max_steps = 800
    ckpt_dir = glob.glob(os.path.join(task+'_ckpt', '*.pt'))[-1]  # last ckpt

    env = Rocket(task=task, max_steps=max_steps)
    net = ActorCritic(input_dim=env.state_dims, output_dim=env.action_dims).to(device)
    if os.path.exists(ckpt_dir):
        checkpoint = torch.load(ckpt_dir)
        net.load_state_dict(checkpoint['model_G_state_dict'])

    state = env.reset()
    for step_id in range(max_steps):
        action, log_prob, value = net.get_action(state)
        state, reward, done, _ = env.step(action)
        env.render(window_name='test')
        if env.already_crash:
            break
```





}
``````
