# from mlagents_envs.environment import UnityEnvironment
# from mlagents_envs.base_env import ActionTuple

# # Path to your Unity executable
# env_path = "/Users/anooprehman/Documents/uoft/extracurricular/design_teams/utmist2/bodiesTEST3/unity_projects/engine 2/Builds/v3testBuild2_bigVec.app"

# try:
#     # Launch the environment
#     env = UnityEnvironment(file_name=env_path)

#     # Reset the environment
#     env.reset()
#     behavior_name = list(env.behavior_specs)[0]
#     spec = env.behavior_specs[behavior_name]

#     decision_steps, terminal_steps = env.get_steps(behavior_name)

#     # # Example of taking an action
#     # action = spec.create_empty_action(len(decision_steps))
#     # action_tuple = ActionTuple(continuous=action)
#     # env.set_actions(behavior_name, action_tuple)

#     env.step()  # Advance the environment by one step

#     # Getting the Observation Vector
#     # decision_steps, terminal_steps = env.get_steps(behavior_name)
#     # for obs in decision_steps.obs:
#     #     print(obs)  # This will print the observation vectors for each agent

#     decision_steps, terminal_steps = env.get_steps(behavior_name)
#     for agent_id in decision_steps:
#         obs = decision_steps[agent_id].obs
#         print(obs)  # This should print the observation vector for each agent

#     # Don't forget to close the environment
#     env.close()
# except Exception as e:
#     print(e)



from mlagents_envs.environment import UnityEnvironment

# Path to your Unity executable
env_path = "/Users/anooprehman/Documents/uoft/extracurricular/design_teams/utmist2/bodiesTEST3/unity_projects/engine 2/Builds/v3testBuild2_bigVec.app"

try:
    # Launch the environment
    env = UnityEnvironment(file_name=env_path)

    # Reset the environment
    env.reset()
    behavior_name = list(env.behavior_specs)[0]
    spec = env.behavior_specs[behavior_name]

    # Step the environment
    env.step()

    # Getting the Observation Vector for each agent
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    for agent_id in decision_steps:
        # Accessing observations for this agent
        observations = decision_steps[agent_id].obs

        # Print or process the observations
        # Note: If you have multiple observation types (vector, visual), they will be in separate elements of the 'observations' list
        print(f"Observations for agent {agent_id}:")
        for obs_idx, obs in enumerate(observations):
            print(f"  Observation space {obs_idx}: {obs}")

    # Don't forget to close the environment
    env.close()
except Exception as e:
    print(f"An error occurred: {e}")