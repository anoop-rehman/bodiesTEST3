from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np

def take_random_actions(env, behavior_name, n_agents, action_spec):
    """Generate and set random actions for agents based on action_spec."""
    branch_sizes = action_spec.discrete_branches
    random_actions = np.column_stack([np.random.randint(0, branch, size=n_agents) for branch in branch_sizes])
    env.set_actions(behavior_name, ActionTuple(discrete=random_actions))

# Load and reset the environment
env = UnityEnvironment(file_name="/Users/andrewgordienko/Documents/env9")
env.reset()

behavior_names = list(env.behavior_specs)

for episode in range(10):
    decision_steps, terminal_steps = {}, {}
    
    # Initialize terminal_steps with non-terminal values
    for behavior_name in behavior_names:
        decision_steps[behavior_name], terminal_steps[behavior_name] = env.get_steps(behavior_name)
    
    steps = 0
    print(episode)
    while steps < 100:
        for behavior_name in behavior_names:
            n_agents = len(decision_steps[behavior_name])
            action_spec = env.behavior_specs[behavior_name].action_spec
            
            take_random_actions(env, behavior_name, n_agents, action_spec)

            # Print observations for agent 0 under specified behavior
            if behavior_name == "My Behavior?team=0" and 0 in decision_steps[behavior_name].agent_id:
                obs_index = list(decision_steps[behavior_name].agent_id).index(0)
                #print(f"Observations for agent 0 under behavior {behavior_name}: {decision_steps[behavior_name].obs[0][obs_index]}")

        env.step()
        steps += 1

        # Update decision and terminal steps
        for behavior_name in behavior_names:
            decision_steps[behavior_name], terminal_steps[behavior_name] = env.get_steps(behavior_name)

    # Reset environment at the end of each episode
    env.reset()

env.close()
