import numpy as np
import xml.etree.ElementTree as ET
from dm_control import mujoco, viewer
from dm_env import specs, TimeStep, StepType
import random
import torch
from ai_components import Network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import numpy 
from torch.distributions.categorical import Categorical

# Define joint ranges and motor gears for increased flexibility and strength
joint_ranges = {
    'hip': '-90 90',
    'knee': '-90 90',
    'ankle': '-50 50'  # New ankle joint range
}
motor_gears = {
    'hip': 200,
    'knee': 200,
    'ankle': 200  # New gear for ankle motor
}

# Lower damping values for more fluid movement
joint_damping = {
    'hip': '2.0',
    'knee': '4.0',
    'ankle': '6.0'  # New damping value for ankle joint
}


class Torso:
    def __init__(self, name="torso", position=(0, 0, 0.75), size=None):
        self.name = name
        self.position = position
        self.size = size if size else (random.uniform(0.2, 0.5), random.uniform(0.1, 0.2), random.uniform(0.05, 0.1))

    def to_xml(self, layer, color):
        torso = ET.Element('body', attrib={'name': self.name, 'pos': ' '.join(map(str, self.position))})
        ET.SubElement(torso, 'geom', attrib={
            'name': f'torso_geom_{self.name}', 
            'type': 'box', 
            'size': ' '.join(map(str, self.size)), 
            'pos': '0 0 0', 
            'contype': '1', 
            'conaffinity': str(layer),
            'material': color  # Assign unique color
        })
        
        ET.SubElement(torso, 'joint', attrib={
            'name': f'{self.name}_root', 
            'type': 'free', 
            'armature': '0', 
            'damping': '0', 
            'limited': 'false'
        })
        ET.SubElement(torso, 'site', attrib={
            'name': f'{self.name}_site', 
            'pos': '0 0 0', 
            'type': 'sphere', 
            'size': '0.01'
        })

        return torso


class Leg:
    def __init__(self, name, torso_size, size):
        self.name = name
        self.torso_size = torso_size
        self.size = size
        self.subparts = 0

    def to_xml(self):
        # Random edge selection for leg placement
        edge_positions = [
            (0, self.torso_size[1]/2, 0),  # Right side
            (0, -self.torso_size[1]/2, 0),  # Left side
            (self.torso_size[0]/2, 0, 0),  # Front side
            (-self.torso_size[0]/2, 0, 0)  # Back side
        ]
        position = random.choice(edge_positions)

        leg = ET.Element('body', attrib={'name': self.name, 'pos': ' '.join(map(str, position))})

        # Random lengths for each part with a small overlap
        upper_length = np.random.uniform(0.1, 0.2)
        lower_length = np.random.uniform(0.1, 0.2)
        foot_length = np.random.uniform(0.1, 0.2)

        # Upper part
        upper_fromto = [0.0, 0.0, 0.0, upper_length, 0.0, 0.0]
        ET.SubElement(leg, 'geom', attrib={'name': self.name + '_upper_geom', 'type': 'capsule', 'fromto': ' '.join(map(str, upper_fromto)), 'size': str(self.size)})
        ET.SubElement(leg, 'joint', attrib={'name': self.name + '_hip_joint', 'type': 'ball', 'damping': joint_damping['hip']})

        # Lower part
        lower_fromto = [upper_length, 0.0, 0.0, upper_length + lower_length, 0.0, 0.0]
        lower_part = ET.SubElement(leg, 'body', attrib={'name': self.name + '_lower', 'pos': ' '.join(map(str, [upper_length, 0.0, 0.0]))})
        ET.SubElement(lower_part, 'geom', attrib={'name': self.name + '_lower_geom', 'type': 'capsule', 'fromto': ' '.join(map(str, lower_fromto)), 'size': str(self.size)})

        # Knee joint
        ET.SubElement(lower_part, 'joint', attrib={'name': self.name + '_knee_joint', 'type': 'hinge', 'axis': '0 1 0', 'range': joint_ranges['knee'], 'damping': joint_damping['knee'], 'limited': 'true'})

        # Foot part
        foot_fromto = [upper_length + lower_length, 0.0, 0.0, upper_length + lower_length + foot_length, 0.0, 0.0]
        foot_part = ET.SubElement(lower_part, 'body', attrib={'name': self.name + '_foot', 'pos': ' '.join(map(str, [upper_length + lower_length, 0.0, 0.0]))})
        ET.SubElement(foot_part, 'geom', attrib={'name': self.name + '_foot_geom', 'type': 'cylinder', 'fromto': ' '.join(map(str, foot_fromto)), 'size': str(self.size)})
        ET.SubElement(foot_part, 'joint', attrib={'name': self.name + '_ankle_joint', 'type': 'ball', 'damping': joint_damping['ankle']})

        self.subparts = 1  # upper part
        self.subparts += 1 if lower_length > 0 else 0
        self.subparts += 1 if foot_length > 0 else 0

        return leg, self.name + '_ankle_joint'

def create_assets_xml():
    assets = ET.Element('asset')

    # Add a checkered texture for the floor
    ET.SubElement(assets, 'texture', attrib={
        'name': 'checkered',
        'type': '2d',
        'builtin': 'checker',
        'rgb1': '0.2 0.3 0.4',  # Color 1 of the checker pattern
        'rgb2': '0.9 0.9 0.9',  # Color 2 of the checker pattern
        'width': '512',         # Texture width
        'height': '512'         # Texture height
    })

    # Add a material that uses the checkered texture
    ET.SubElement(assets, 'material', attrib={
        'name': 'MatCheckered',
        'texture': 'checkered',
        'reflectance': '0.5'
    })

    # Material for the plane (floor)
    ET.SubElement(assets, 'material', attrib={
        'name': 'MatPlane',
        'reflectance': '0.5',
        'shininess': '1',
        'specular': '1'
    })

    # Define materials for the colors used for creatures and flags
    color_materials = {
        'red': '1 0 0 1', 
        'green': '0 1 0 1', 
        'blue': '0 0 1 1',
        'yellow': '1 1 0 1', 
        'purple': '0.5 0 0.5 1', 
        'orange': '1 0.5 0 1',
        'pink': '1 0.7 0.7 1', 
        'grey': '0.5 0.5 0.5 1', 
        'brown': '0.6 0.3 0 1'
    }

    for name, rgba in color_materials.items():
        ET.SubElement(assets, 'material', attrib={'name': name, 'rgba': rgba})

    return assets



def create_floor_xml(size=(10, 10, 0.1)):
    return ET.Element('geom', attrib={'name': 'floor', 'type': 'plane', 'size': ' '.join(map(str, size)), 'pos': '0 0 0', 'material': 'MatCheckered'})

def create_flag_xml(flag_id, layer, color, floor_size=(10, 10, 0.1)):
    flag_x = random.uniform(-floor_size[0]/2, floor_size[0]/2)
    flag_y = random.uniform(-floor_size[1]/2, floor_size[1]/2)
    flag_z = 0  # On the floor
    flag_position = (flag_x, flag_y, flag_z)

    flag_size = (0.05, 0.05, 0.5)  # Cube size
    return ET.Element('geom', attrib={
        'name': f'flag_{flag_id}', 
        'type': 'box', 
        'size': ' '.join(map(str, flag_size)), 
        'pos': ' '.join(map(str, flag_position)), 
        'material': color,  # Assign the color material
        'contype': '1', 
        'conaffinity': str(layer)
    })

class CustomAntEnv:
    def __init__(self, xml_string, leg_info, max_steps=2500, num_creatures=9):
        self.xml_string = xml_string
        self.leg_info = leg_info
        self.max_steps = max_steps
        self.step_count = 0
        self.num_creatures = num_creatures

        # Initialize previous_velocities later
        self.previous_velocities = None

        # Now call reset
        self.reset()

    def reset(self):
        self.physics = mujoco.Physics.from_xml_string(self.xml_string)

        if self.previous_velocities is None:
            MAX_LEGS = 4
            SUBPARTS_PER_LEG = 3
            self.previous_velocities = np.zeros(MAX_LEGS * SUBPARTS_PER_LEG * self.physics.model.nv // self.num_creatures)
            
        self.step_count = 0
        self.flag_positions = [self._get_flag_position(i) for i in range(self.num_creatures)]
        return TimeStep(StepType.FIRST, reward=0.0, discount=1.0, observation=self._get_observation())

    def step(self, action):
        # Flatten the action if it's a 2D array (for multiple creatures)
        if len(action.shape) > 1:
            action = action.flatten()

        # Process and set control for each creature
        for creature_id in range(self.num_creatures):
            # Calculate the start and end indices for this creature's actions
            start_idx = creature_id * self.physics.model.nu // self.num_creatures
            end_idx = (creature_id + 1) * self.physics.model.nu // self.num_creatures
            creature_action = action[start_idx:end_idx]
            
            # Set control for this creature's motors
            self.physics.data.ctrl[start_idx:end_idx] = creature_action

        # Advance the simulation
        self.physics.step()
        self.step_count += 1

        # Calculate rewards and observation
        rewards = [self._calculate_reward(i) for i in range(self.num_creatures)]
        observation = self._get_observation()
        done = self._is_done()
        step_type = StepType.LAST if done else StepType.MID

        # Sum the rewards if they are per creature, or use the total reward
        total_reward = np.sum(rewards) if isinstance(rewards, list) else rewards

        return TimeStep(step_type, rewards, discount=1.0, observation=observation)

    def _process_action_for_creature(self, action, creature_id):
        num_limbs = self._get_num_limbs_for_creature(creature_id)
        creature_specific_actions = action[creature_id * 12 : (creature_id + 1) * 12]  # Extract 12 actions for this creature

        # Initialize an array to store processed actions for this creature's motors
        creature_action = np.zeros(self.physics.model.nu // self.num_creatures)

        action_idx = 0  # Index for actions in creature_specific_actions
        motor_idx = 0  # Index to map to the correct motors
        for limb_id in range(num_limbs):
            num_subparts = self.leg_info[creature_id][limb_id]

            for subpart_id in range(num_subparts):
                start_idx = action_idx * 3
                end_idx = start_idx + 3

                if start_idx < len(creature_specific_actions):
                    subpart_action = creature_specific_actions[start_idx:end_idx]

                    # Assuming each subpart corresponds to one motor in sequence
                    creature_action[motor_idx : motor_idx + 3] = subpart_action
                    motor_idx += 3  # Increment to the next set of motors

                action_idx += 1  # Move to the next set of actions

                if action_idx >= 4:  # Limit to 4 limbs (12 actions / 3 actions per limb)
                    break

            if action_idx >= 4:  # Exit outer loop if all actions have been assigned
                break

        return creature_action

    def _get_num_limbs_for_creature(self, creature_id):
        return len(self.leg_info[creature_id])

    def _calculate_reward(self, creature_id):
        # Stability Reward (Lower priority)
        upright_orientation = self._get_upright_orientation(creature_id)
        stability_reward = np.exp(-10 * np.abs(1 - upright_orientation))
        stability_reward = np.clip(stability_reward, 0, 1)

        # Energy Efficiency (Second priority)
        energy_penalty = np.sum(np.abs(self.physics.data.ctrl)) * 0.00005  # Adjusted penalty scaling

        # Goal-Directed Behavior (Highest priority)
        distance = np.linalg.norm(self._get_torso_position(creature_id) - self.flag_positions[creature_id])
        if distance < 0.1:
            distance_reward = 10  # High reward for reaching the flag
        else:
            distance_reward = np.clip(1 / (1 + distance), 0, 1)  # Gradual reward for approaching the flag

        # Total Reward Calculation
        total_reward = distance_reward + stability_reward - energy_penalty
        return total_reward


    def _get_upright_orientation(self, creature_id):
        # Calculate the orientation of the torso with respect to the vertical axis
        # This is a simplified example; you may need to adjust it based on your model
        z_axis = self.physics.named.data.geom_xmat[f'torso_geom_torso_{creature_id}'][2::3]
        vertical_alignment = np.dot(z_axis, np.array([0, 0, 1]))  # Dot product with vertical axis
        return vertical_alignment


    def _get_observation(self):
        # Constants for observation sizes
        MAX_LEGS = 4
        SUBPARTS_PER_LEG = 3
        DATA_POINTS_PER_SUBPART = 3
        DISTANCE_TO_TARGET_DIMS = 2  # Distance to target (x, y)

        # Initialize a list to store observations of all creatures
        all_creature_observations = []

        # Iterate through each creature
        for creature_id in range(self.num_creatures):
            # Initialize observation arrays for this creature
            leg_observation = np.zeros(MAX_LEGS * SUBPARTS_PER_LEG * DATA_POINTS_PER_SUBPART)
            distance_to_target = np.zeros(DISTANCE_TO_TARGET_DIMS)

            # Calculate leg observations for this creature
            for i in range(MAX_LEGS):
                if i < len(self.leg_info[creature_id]):
                    for j in range(SUBPARTS_PER_LEG):
                        if j < self.leg_info[creature_id][i]:
                            physics_idx = i * SUBPARTS_PER_LEG + j
                            angle = self.physics.data.qpos[physics_idx]
                            velocity = self.physics.data.qvel[physics_idx]
                            acceleration = (velocity - self.previous_velocities[physics_idx]) / 0.02
                            self.previous_velocities[physics_idx] = velocity
                            obs_idx = i * (SUBPARTS_PER_LEG * DATA_POINTS_PER_SUBPART) + j * DATA_POINTS_PER_SUBPART
                            leg_observation[obs_idx:obs_idx+3] = [angle, velocity, acceleration]

            # Calculate distance to target for this creature
            distance_to_target = self._calculate_distance_to_target(creature_id)

            # Combine observations for this creature into a single array
            creature_observation = np.concatenate([leg_observation, distance_to_target])

            # Add this creature's observation to the list
            all_creature_observations.append(creature_observation)

        # Concatenate all creatures' observations into one array
        all_observations = np.concatenate(all_creature_observations)
        return all_observations


    def _calculate_distance_to_target(self, creature_id):
        # Calculate distance to the target stick for the specified creature
        torso_position = self._get_torso_position(creature_id)
        flag_position = self._get_flag_position(creature_id)
        distance = flag_position[:2] - torso_position[:2]  # Only consider x and y coordinates
        return distance
    
    def _is_done(self):
        if self.step_count >= self.max_steps:
            return True
        for creature_id in range(self.num_creatures):
            distance = np.linalg.norm(self._get_torso_position(creature_id) - self.flag_positions[creature_id])
            if distance < 0.1:
                return True
        return False

    def _get_torso_position(self, creature_id):
        return self.physics.named.data.geom_xpos[f'torso_geom_torso_{creature_id}']

    def _get_flag_position(self, flag_id):
        return self.physics.named.data.geom_xpos[f'flag_{flag_id}']

    def action_spec(self):
        action_shape = (self.num_creatures, 12)  # Define the shape of the action array
        minimum = -1  # Define the minimum value of each action
        maximum = 1   # Define the maximum value of each action
        return specs.BoundedArray(shape=action_shape, dtype=np.float32, minimum=minimum, maximum=maximum)

    def observation_spec(self):
        observation_shape = (342,)  # 36 data points per creature
        return specs.Array(shape=observation_shape, dtype=np.float32)


def create_ant_model(num_creatures=9):
    mujoco_model = ET.Element('mujoco')
    mujoco_model.append(create_assets_xml())
    worldbody = ET.SubElement(mujoco_model, 'worldbody')
    worldbody.append(create_floor_xml(size=(10, 10, 0.1)))

    actuator = ET.SubElement(mujoco_model, 'actuator')

    # Define a list of colors for creatures and flags
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'grey', 'brown']  # Add more colors if needed

    creature_leg_info = {}  # Dictionary to store leg and subpart info

    for creature_id in range(num_creatures):
        layer = creature_id + 1
        color = colors[creature_id % len(colors)]

        # Adjust the initial position to spread out the creatures
        initial_position = (creature_id - num_creatures / 2, 0, 0.75)
        
        torso_obj = Torso(name=f'torso_{creature_id}', position=initial_position)
        torso_xml = torso_obj.to_xml(layer, color)
        worldbody.append(torso_xml)

        # Create a flag with the same color as the torso
        worldbody.append(create_flag_xml(creature_id, layer, color))

        num_legs = random.randint(1, 4)
        leg_size = 0.04
        leg_info = []

        for i in range(num_legs):
            leg_name = f"leg_{creature_id}_{i+1}"

            # Create Leg object with random edge placement
            leg_obj = Leg(leg_name, torso_obj.size, leg_size)
            leg_xml, foot_joint_name = leg_obj.to_xml()
            torso_xml.append(leg_xml)

            # Add motors for each joint
            ET.SubElement(actuator, 'motor', attrib={
                'name': f'{leg_name}_hip_motor',
                'joint': f'{leg_name}_hip_joint',
                'ctrllimited': 'true',
                'ctrlrange': '-1 1',
                'gear': str(motor_gears['hip'])
            })

            # Add motors for knee and ankle if they exist
            if 'knee_joint' in foot_joint_name:
                ET.SubElement(actuator, 'motor', attrib={
                    'name': f'{leg_name}_knee_motor',
                    'joint': f'{leg_name}_knee_joint',
                    'ctrllimited': 'true',
                    'ctrlrange': '-1 1',
                    'gear': str(motor_gears['knee'])
                })

            if 'ankle_joint' in foot_joint_name:
                ET.SubElement(actuator, 'motor', attrib={
                    'name': f'{foot_joint_name}_motor',
                    'joint': foot_joint_name,
                    'ctrllimited': 'true',
                    'ctrlrange': '-1 1',
                    'gear': str(motor_gears['ankle'])
                })
            
            leg_info.append(leg_obj.subparts)
        
        creature_leg_info[creature_id] = leg_info

    # Add sensors
    sensors = ET.SubElement(mujoco_model, 'sensor')
    for creature_id in range(num_creatures):
        torso_name = f'torso_{creature_id}'
        ET.SubElement(sensors, 'accelerometer', attrib={'name': f'{torso_name}_accel', 'site': f'{torso_name}_site'})
        ET.SubElement(sensors, 'gyro', attrib={'name': f'{torso_name}_gyro', 'site': f'{torso_name}_site'})

    xml_string = ET.tostring(mujoco_model, encoding='unicode')
    return xml_string, creature_leg_info


# Instantiate the network

def network_policy(time_step):
    if not time_step.first():
        observation_tensor = torch.tensor(time_step.observation, dtype=torch.float32).unsqueeze(0)
        action = network(observation_tensor)
        action_np = action.view(env.num_creatures, -1).detach().numpy()
        action_np = np.clip(action_np, env.action_spec().minimum, env.action_spec().maximum)

        # Check for NaN or Inf in network output
        if np.any(np.isnan(action_np)) or np.any(np.isinf(action_np)):
            print("Warning: NaN or Inf in network output.")
            action_np = np.zeros_like(action_np)

        # Scale down the actions here if needed
        scaled_action_np = np.tanh(action_np)  # Using tanh to normalize actions between -1 and 1

        return scaled_action_np
    else:
        return np.zeros(env.action_spec().shape)


def random_policy(time_step):
    action_spec = env.action_spec()
    return np.random.uniform(action_spec.minimum, action_spec.maximum, size=action_spec.shape)

# Create the environment
xml_string, leg_info = create_ant_model()
env = CustomAntEnv(xml_string, leg_info)

# Initialize network with the correct input shape
input_shape = (env.observation_spec().shape[0],)
network = Network(input_shape)

input_shape = (env.observation_spec().shape[0],)
action_dim = env.action_spec().shape[1]  # Adjust as necessary
print(input_shape)
print(action_dim)

# Launch the viewer with the environment and the custom policy
#viewer.launch(env, policy=network_policy)

EPISODES = 501
MEM_SIZE = 1000000
BATCH_SIZE = 5
GAMMA = 0.95
EXPLORATION_MAX = 1.0
EXPLORATION_DECAY = 0.999
EXPLORATION_MIN = 0.001
LEARNING_RATE = 0.0003
FC1_DIMS = 1024
FC2_DIMS = 512
ENTROPY_BETA = 0.02  # This is the hyperparameter that controls the strength of the entropy regularization. You might need to tune it.
DEVICE = torch.device("cpu")

best_reward = float("-inf")
average_reward = 0
episode_number = []
average_reward_number = []
    
class actor_network(nn.Module):
    def __init__(self, input_shape, action_space):
        super().__init__()

        self.std = 0.5

        self.fc1 = nn.Linear(input_shape[0], FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, action_space)

        self.log_std = nn.Parameter(torch.ones(1, action_space) * 0.01)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

        self.log_std_min = -20  # min bound for log standard deviation
        self.log_std_max = 2    # max bound for log standard deviation


    def net(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))   # Use tanh for the last layer

        return x
    
    def forward(self, x):
        mu = self.net(x)
        # Clipping the log std deviation between predefined min and max values
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp().expand_as(mu)
        policy_dist = torch.distributions.Normal(mu, std)
        return policy_dist

class critic_network(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        # Unpack the input_shape tuple when passing it to nn.Linear
        self.fc1 = nn.Linear(*input_shape, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.to(DEVICE)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class Agent:
    def __init__(self, n_actions, input_dims):
        self.gamma = 0.99
        self.policy_clip = 0.2
        self.n_epochs = 10
        self.gae_lambda = 0.95

        # Use input_dims directly
        single_creature_obs_shape = (input_dims[0] // env.num_creatures,)

        self.actor = actor_network(single_creature_obs_shape, n_actions)
        self.critic = critic_network(single_creature_obs_shape)  # Updated line
        self.memory = PPOMemory(BATCH_SIZE)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(DEVICE)
        
        # directly assign the Normal distribution object to dist
        dist = self.actor(state)
        
        mu = dist.mean
        sigma = dist.stddev
        
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        value = self.critic.forward(state)
        
        action = action.cpu().numpy()[0]
        log_prob = log_prob.item()
        value = torch.squeeze(value).item()

        return action, log_prob, value


    def learn(self):
        for i in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1] * (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            
            advantage = torch.tensor(advantage).to(DEVICE)
            values = torch.tensor(values).to(DEVICE)

            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(DEVICE)
                old_probs = torch.tensor(old_prob_arr[batch]).to(DEVICE)
                actions = torch.tensor(action_arr[batch]).to(DEVICE)

                dist = self.actor(states)  # Get the policy distribution
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)

                new_probabilities = dist.log_prob(actions).sum(axis=-1)

                probability_ratio = new_probabilities.exp() / old_probs.exp()

                weighted_probabilities = advantage[batch] * probability_ratio
                weighted_clipped_probabilities = torch.clamp(probability_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]
                
                entropy = dist.entropy().mean()
                actor_loss = -torch.min(weighted_probabilities, weighted_clipped_probabilities).mean() - ENTROPY_BETA * entropy

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        
        self.memory.clear_memory()  

# Assuming you have already defined classes: Agent, actor_network, critic_network, etc.

# Initialize the custom environment
env = CustomAntEnv(xml_string, leg_info)

# Initialize the network
#network = Network(env.observation_spec().shape[0])

# Adjust the agent initialization to match your environment
input_dims = env.observation_spec().shape  # Now this is correct
agent = Agent(n_actions=env.action_spec().shape[1], input_dims=input_dims)

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

