import numpy as np
from dm_control import mujoco, viewer
from environment_class import CustomAntEnv
from asset_components import create_ant_model
from ppo_agent import Agent

xml_string, leg_info = create_ant_model()
env = CustomAntEnv(xml_string, leg_info)

input_dims = env.observation_spec().shape  # Now this is correct
agent = Agent(n_actions=env.action_spec().shape[1], input_dims=input_dims, env=env)

import matplotlib.pyplot as plt
import os

# Directory for saving plots
plot_dir = "saved_plots"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

def save_initial_frame(env, episode, status):
    image = env.physics.render()
    plt.imshow(image)
    plt.title(f"Initial Frame of Episode {episode}")

    # Save the plot instead of displaying it
    plt.savefig(f"{plot_dir}/episode_{episode}_{status}.png")
    plt.close()  # Close the plot to free up memory

print("training started")
for episode in range(1, 20):
    xml_string, leg_info = create_ant_model()
    env = CustomAntEnv(xml_string, leg_info)
    timestep = env.reset()
    score = 0
    done = False
    step = 0
    total_reward = np.zeros((1, env.num_creatures))

    save_initial_frame(env, episode, "initial")

    while not done:
        actions = []
        log_probs = []
        values = []
        rewards = []
        dones = []

        for creature_id in range(env.num_creatures):
            # Extract the observation for the current creature
            start_idx = creature_id * 38  # Each creature has 38 observations
            end_idx = start_idx + 38
            observation = timestep.observation[start_idx:end_idx]

            action, log_prob, value = agent.choose_action(observation)
            
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
        
        # Combine actions for all creatures and perform a step in the environment
        combined_actions = np.concatenate(actions)
        timestep = env.step(combined_actions)

        # Update rewards, dones, etc., for each creature
        for creature_id in range(env.num_creatures):
            reward = timestep.reward[creature_id]
            total_reward[0][creature_id] += reward
            done = timestep.last()
            score += reward

            # Again, extract the observation for the current creature
            observation = timestep.observation[creature_id * 38:(creature_id + 1) * 38]
            agent.remember(observation, actions[creature_id], log_probs[creature_id], values[creature_id], reward, done)

        if step % 20 == 0:
            agent.learn()
            print("learn")

        step += 1

        if done:
            # Process end of episode
            save_initial_frame(env, episode, "end")
            break
    
    print(episode, total_reward/1000)

def trained_policy(time_step):
    if not time_step.first():
        # Extract observations for all creatures
        observations = time_step.observation
        actions = []

        for creature_id in range(env.num_creatures):
            start_idx = creature_id * 38  # Adjust this if the observation size per creature is different
            end_idx = start_idx + 38
            observation = observations[start_idx:end_idx]

            action, _, _ = agent.choose_action(observation)
            actions.append(action)

        # Combine actions for all creatures
        combined_actions = np.concatenate(actions)
        return combined_actions
    else:
        return np.zeros(env.action_spec().shape)

# Launch the viewer with the environment and the trained policy
viewer.launch(env, policy=trained_policy)
