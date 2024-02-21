from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

# Path to your Unity executable
env_path = "/Users/anooprehman/Documents/uoft/extracurricular/design_teams/utmist2/bodiesTEST3/unity_projects/engine 2/Builds/v3testBuild2_bigVec.app"

try:
    # Launch the environment
    env = UnityEnvironment(file_name=env_path)

    # Reset the environment
    env.reset()
    behavior_name = list(env.behavior_specs)[0]
    spec = env.behavior_specs[behavior_name]

    decision_steps, terminal_steps = env.get_steps(behavior_name)

    # # Example of taking an action
    # action = spec.create_empty_action(len(decision_steps))
    # action_tuple = ActionTuple(continuous=action)
    # env.set_actions(behavior_name, action_tuple)

    env.step()  # Advance the environment by one step

    # Getting the Observation Vector
    # decision_steps, terminal_steps = env.get_steps(behavior_name)
    # for obs in decision_steps.obs:
    #     print(obs)  # This will print the observation vectors for each agent

    decision_steps, terminal_steps = env.get_steps(behavior_name)
    for agent_id in decision_steps:
        obs = decision_steps[agent_id].obs
        print(obs)  # This should print the observation vector for each agent

    # Don't forget to close the environment
    env.close()
except Exception as e:
    print(e)