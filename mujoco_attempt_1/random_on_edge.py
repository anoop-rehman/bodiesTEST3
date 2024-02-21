import numpy as np
from dm_control import mujoco
from dm_control import suite, viewer
from dm_env import specs, TimeStep, StepType
from lxml import etree
import random

# Load the XML file
xml_path = 'path_to_your_mujoco_xml_file.xml'
with open(xml_path, 'r') as file:
    xml_string = file.read()

def randomize_model(xml_string):
    root = etree.fromstring(xml_string)

    # Randomize torso size
    torso_geom = root.find(".//geom[@name='torso_geom']")
    torso_size = np.random.uniform(0.1, 0.5, size=3)  # (length, width, height)
    if torso_geom is not None:
        torso_geom.set('size', ' '.join(map(str, torso_size)))
        torso_geom.set('type', 'box')  # Change the torso to a box

    # Randomize the number of legs (1 to 4)
    num_legs = random.randint(1, 4)
    leg_names = ['front_left_leg', 'front_right_leg', 'back_leg', 'right_back_leg']
    random.shuffle(leg_names)
    legs_to_keep = leg_names[:num_legs]

    body = root.find(".//body[@name='torso']")
    actuators = root.find(".//actuator")

    # Define possible vertices based on the torso size
    vertices = [
        (torso_size[0] / 2, torso_size[1] / 2, torso_size[2] / 2),
        (-torso_size[0] / 2, torso_size[1] / 2, torso_size[2] / 2),
        (-torso_size[0] / 2, -torso_size[1] / 2, torso_size[2] / 2),
        (torso_size[0] / 2, -torso_size[1] / 2, torso_size[2] / 2)
    ]

    if body is not None:
        # Attach legs to random vertices and remove unwanted legs
        for i, leg_name in enumerate(leg_names):
            leg = body.find(f".//body[@name='{leg_name}']")
            if leg_name in legs_to_keep and leg is not None:
                vertex = vertices[i]
                leg.set('pos', ' '.join(map(str, vertex)))
            elif leg is not None:
                body.remove(leg)

        # Update actuators based on the legs present
        for actuator in actuators.findall(".//motor"):
            joint_name = actuator.get('joint')
            if any(leg_name in joint_name for leg_name in legs_to_keep):
                continue
            actuators.remove(actuator)

    return etree.tostring(root, pretty_print=True).decode()


def modify_leg_structure(leg):
    num_subparts_to_remove = random.randint(1, 2)
    print(num_subparts_to_remove)
    subparts = [child for child in leg if child.tag == 'body']
    removed_joints = set()

    for _ in range(min(num_subparts_to_remove, len(subparts))):
        subpart_to_remove = random.choice(subparts)
        for joint in subpart_to_remove.findall('.//joint'):
            removed_joints.add(joint.get('name'))
        leg.remove(subpart_to_remove)
        subparts.remove(subpart_to_remove)

    return removed_joints


# Custom environment class
class CustomAntEnv:

    def __init__(self, xml_string):
        self.xml_string = xml_string
        self.physics = None
        self.reset()

    def reset(self):
        modified_xml = randomize_model(self.xml_string)
        self.physics = mujoco.Physics.from_xml_string(modified_xml)
        # Additional reset steps if necessary

    def step(self, action):
        self.physics.set_control(action)
        self.physics.step()

        # Compute reward and observation
        reward = self._calculate_reward()
        observation = self._get_observation()

        # Check if the episode is over
        done = self._is_done()

        # Create and return the time_step
        if done:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID
        return TimeStep(step_type, reward, self.physics.model.opt.timestep, observation)

    def _calculate_reward(self):
        return 0.0  # Implement reward calculation

    def _get_observation(self):
        return np.zeros(self.physics.model.nq + self.physics.model.nv)  # Implement observation calculation

    def _is_done(self):
        return False  # Implement termination condition

    def action_spec(self):
        action_size = self.physics.model.nu
        return specs.BoundedArray(shape=(action_size,), dtype=np.float32, minimum=-1, maximum=1)

    def observation_spec(self):
        observation_shape = self.physics.model.nq + self.physics.model.nv
        return specs.Array(shape=(observation_shape,), dtype=np.float32)

# Initialize the custom environment
env = CustomAntEnv(xml_string)

def random_policy(time_step):
    # Random action policy
    action_spec = env.action_spec()
    action = np.random.uniform(action_spec.minimum, action_spec.maximum, size=action_spec.shape)
    return action

# Launch the viewer with the random policy
viewer.launch(env, random_policy)
