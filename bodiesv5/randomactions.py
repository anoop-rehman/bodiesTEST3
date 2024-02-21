# this code plays out random actions for all four players

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np

# Load the environment
env = UnityEnvironment(file_name="/Users/andrewgordienko/Documents/soccerenvv1")

# Reset the environment to get the behavior names
env.reset()
behavior_names = list(env.behavior_specs)

for episode in range(10):  # Play 10 episodes
    env.reset()
    
    decision_steps, terminal_steps = {}, {}

    for behavior_name in behavior_names:
        decision_steps[behavior_name], terminal_steps[behavior_name] = env.get_steps(behavior_name)

    while not all(terminal_steps.values()):  # Play until the episode ends for all agents
        for behavior_name in behavior_names:
            # Get the number of agents and the action size
            n_agents = len(decision_steps[behavior_name])
            action_size = env.behavior_specs[behavior_name].action_spec.discrete_branches

            # Generate random discrete actions
            random_actions = np.column_stack([np.random.randint(0, branch, size=n_agents) for branch in action_size])

            # Create the ActionTuple
            action_tuple = ActionTuple(continuous=np.zeros((n_agents, 0)), discrete=random_actions)

            # Set the actions
            env.set_actions(behavior_name, action_tuple)

        # Step the environment once after setting actions for all agents
        env.step()

        # Get the new decision and terminal steps for all agents
        for behavior_name in behavior_names:
            decision_steps[behavior_name], terminal_steps[behavior_name] = env.get_steps(behavior_name)

env.close()
