import numpy as np
from dm_control import viewer
from dm_control.locomotion import soccer as dm_soccer
from copy import deepcopy

class SoccerEnv:
    def __init__(self):
        self.ACTION_MAPPINGS = { # roll steer kick
            "move_up": np.array([1.0, 0.0, 0.0]),
            "move_down": np.array([-1.0, 0.0, 0.0]),
            "turn_left": np.array([0.0, -1, 0.0]), # i want this to be move left
            "turn_right": np.array([0.0, 1.0, 0.0]),
            "shoot": np.array([0.0, 0.0, 1.0]),
            "no_action": np.array([0.0, 0.0, 0.0]),
        }

        self.env = dm_soccer.load(
            team_size=2,
            time_limit=10.0,
            disable_walker_contacts=False,
            enable_field_box=True,
            terminate_on_goal=False,
            walker_type=dm_soccer.WalkerType.BOXHEAD
        )

        self.action_specs = self.env.action_spec()
    
    def intuitive_to_action(self, intuitive_commands):
        # convert intuitive commands to numerical action arrays
        actions = []
        for command in intuitive_commands:
            base_action = self.ACTION_MAPPINGS[command]
            actions.append(deepcopy(base_action))
        
        return actions
    
    def step(self, intuitive_commands):
        # execute actions and return the state, reward, and done flag
        actions = self.intuitive_to_action(intuitive_commands)
        time_step = self.env.step(actions)

        relative_player_positions, relative_ball_position, player_possessions = self.get_relative_positions(self.env.physics)
        field_representation = self.get_field_representation(self.env.physics)
        player_orientations = self.get_player_orientations(self.env.physics)

        state = {
            'relative_player_positions': relative_player_positions,
            'relative_ball_position': relative_ball_position,
            'player orientation': player_orientations,
            'player_possessions': player_possessions,
            'field_representation': field_representation
        }
        
        reward = 0  # Placeholder reward
        done = time_step.last()
        
        return state, reward, done
    
    def policy(self, time_step):
        return self.intuitive_to_action(intuitive_commands)

    def visualize(self):
        # visualize environment
        viewer.launch(self.env, policy=self.policy)
    
    def get_relative_positions(self, physics):
        # Retrieve relative positions of players and ball.
        player_names = ["home0/head_body", "home1/head_body", "away0/head_body", "away1/head_body"]
        ball_name = "soccer_ball/"

        player_positions = [physics.named.data.xpos[player][:2].copy() for player in player_names]
        ball_position = physics.named.data.xpos[ball_name][:2].copy()

        center_of_field = np.array([0.0, 0.0])
        relative_player_positions = [pos - center_of_field for pos in player_positions]
        relative_ball_position = ball_position - center_of_field

        possession_threshold = 1.0
        player_possessions = [np.linalg.norm(pos - ball_position) < possession_threshold for pos in relative_player_positions]

        return relative_player_positions, relative_ball_position, player_possessions

    def get_field_representation(self, physics):
        # Generate a grid representation of the field.
        player_size = np.array([0.2, 0.2])
        field_size = np.array([12, 9])
        grid_size = (field_size / player_size).astype(int)
        field_array = np.zeros(grid_size)

        def to_grid_coordinates(pos):
            x = int(pos[0] / player_size[0])
            y = int(pos[1] / player_size[1])
            x = max(0, min(x, grid_size[0] - 1))
            y = max(0, min(y, grid_size[1] - 1))
            return x, y

        for idx, player_name in enumerate(["home0/head_body", "home1/head_body", "away0/head_body", "away1/head_body"]):
            player_pos = physics.named.data.xpos[player_name][:2]
            x, y = to_grid_coordinates(player_pos)
            field_array[x, y] = idx + 1
        
        ball_pos = physics.named.data.xpos["soccer_ball/"][:2]
        x, y = to_grid_coordinates(ball_pos)
        field_array[x, y] = 5
        
        for y in range(field_array.shape[1]):
            field_array[0, y] = 6
            field_array[-1, y] = 6
        
        return field_array
    
    def get_player_orientations(self, physics):
        """Retrieve the orientations (yaw) of players."""
        player_names = ["home0/head_body", "home1/head_body", "away0/head_body", "away1/head_body"]
        
        player_orientations = []
        for player in player_names:
            # Extract quaternion
            quat = physics.named.data.xquat[player]
            
            # Convert quaternion to Euler angles
            _, yaw, _ = self.quaternion_to_euler(quat)
            
            player_orientations.append(yaw)
        
        return player_orientations

    def quaternion_to_euler(self, quat):
        """Convert quaternion to Euler angles."""
        # Extract quaternion components
        w, x, y, z = quat
        
        # Convert to Euler angles
        pitch = np.arcsin(2.0 * (w * y - z * x))
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2))
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
        
        return roll, pitch, yaw

    
    def initialize(self):
        """Initialize the environment and return the initial state."""
        time_step = self.env.reset()
        relative_player_positions, relative_ball_position, player_possessions = self.get_relative_positions(self.env.physics)
        field_representation = self.get_field_representation(self.env.physics)
        
        state = {
            'relative_player_positions': relative_player_positions,
            'relative_ball_position': relative_ball_position,
            'player_possessions': player_possessions,
            'field_representation': field_representation
        }
        
        return state

soccerenv = SoccerEnv()
initial_state = soccerenv.initialize()
done = False

while True:
    intuitive_commands = ["move_up", "move_down", "turn_left", "turn_right"]
    state, reward, done = soccerenv.step(intuitive_commands)
    print("State:", state)
    print("Reward:", reward)
    soccerenv.visualize()
