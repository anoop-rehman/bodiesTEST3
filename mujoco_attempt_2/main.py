import numpy as np
import xml.etree.ElementTree as ET
from dm_control import mujoco, viewer
from dm_env import specs, TimeStep, StepType
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Network(nn.Module):
    def __init__(self, input_shape, num_creatures=9, action_dim_per_creature=12, FC1_DIMS=256, FC2_DIMS=256, LEARNING_RATE=0.001):
        super(Network, self).__init__()
        self.input_shape = input_shape
        self.num_creatures = num_creatures
        self.action_dim_per_creature = action_dim_per_creature

        # Define the layers
        self.fc1 = nn.Linear(*self.input_shape, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, self.num_creatures * self.action_dim_per_creature)

        # Define optimizer and loss function
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()

        # If you have a CUDA device, you can uncomment the next line to use it
        # self.to(DEVICE)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Reshape the output to have the shape (num_creatures, action_dim_per_creature)
        return x.view(-1, self.num_creatures, self.action_dim_per_creature)

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

import numpy as np
from dm_env import TimeStep, StepType
from dm_control import mujoco

class CustomAntEnv:
    def __init__(self, xml_string, max_steps=5000, num_creatures=9):
        self.xml_string = xml_string
        self.physics = None
        self.max_steps = max_steps
        self.step_count = 0
        self.num_creatures = num_creatures
        self.reset()

    def reset(self):
        self.physics = mujoco.Physics.from_xml_string(self.xml_string)
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

        return TimeStep(step_type, total_reward, discount=1.0, observation=observation)

    def _process_action_for_creature(self, action, creature_id):
        # Implement logic to process action for each creature
        # This is a simplified example
        num_limbs = self._get_num_limbs_for_creature(creature_id)
        creature_action = np.zeros(self.physics.model.nu)  # Adjust size as per your model

        for i in range(num_limbs):
            start_idx = i * 3
            end_idx = start_idx + 3
            limb_action = action[start_idx:end_idx]
            # Map limb_action to creature's motors

        return creature_action

    def _get_num_limbs_for_creature(self, creature_id):
        # Return the number of limbs for a given creature
        return 4  # Example: return 4 if all creatures have 4 limbs

    def _calculate_reward(self, creature_id):
        distance = np.linalg.norm(self._get_torso_position(creature_id) - self.flag_positions[creature_id])
        reward = 1 / (1 + distance)
        if distance < 0.1:
            reward += 100
        return reward

    def _get_observation(self):
        # Implement your observation logic here
        # Example: concatenate joint angles and velocities
        joint_angles = self.physics.data.qpos
        joint_velocities = self.physics.data.qvel
        return np.concatenate([joint_angles, joint_velocities])

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
        observation_shape = self.physics.model.nq + self.physics.model.nv
        return specs.Array(shape=(observation_shape,), dtype=np.float32)

# Additional functions for creating and launching the environment may be implemented as needed


def create_ant_model(num_creatures=9):
    mujoco_model = ET.Element('mujoco')
    mujoco_model.append(create_assets_xml())
    worldbody = ET.SubElement(mujoco_model, 'worldbody')
    worldbody.append(create_floor_xml(size=(10, 10, 0.1)))

    actuator = ET.SubElement(mujoco_model, 'actuator')

    # Define a list of colors for creatures and flags
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'grey', 'brown']  # Add more colors if needed

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

    # Add sensors
    sensors = ET.SubElement(mujoco_model, 'sensor')
    for creature_id in range(num_creatures):
        torso_name = f'torso_{creature_id}'
        ET.SubElement(sensors, 'accelerometer', attrib={'name': f'{torso_name}_accel', 'site': f'{torso_name}_site'})
        ET.SubElement(sensors, 'gyro', attrib={'name': f'{torso_name}_gyro', 'site': f'{torso_name}_site'})

    xml_string = ET.tostring(mujoco_model, encoding='unicode')
    return xml_string


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
xml_string = create_ant_model()
env = CustomAntEnv(xml_string)

# Initialize network with the correct input shape
input_shape = (env.observation_spec().shape[0],)
network = Network(input_shape)

# Launch the viewer with the environment and the custom policy
#viewer.launch(env, policy=network_policy)
viewer.launch(env, policy=network_policy)
