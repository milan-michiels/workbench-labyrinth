import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from gymnasium.core import RenderFrame


class LabyrinthEnv(gym.Env):
    """
    Custom Gymnasium environment for the Labyrinth game.
    The environment simulates a ball moving through a maze under the influence of tilting.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, maze_size=(9, 9), wall_thickness=0.5, render_mode='human'):
        super().__init__()

        # Environment parameters
        self.maze_size = maze_size
        self.wall_thickness = wall_thickness
        self.max_tilt_angle = 30.0  # maximum tilt in degrees
        self.dt = 0.05  # increased timestep
        self.friction = 0.2  # reduced friction
        self.render_mode = render_mode
        self.prev_distance = None
        # self.obs_resolution = obs_resolution

        # Ball physics parameters
        self.gravity = 9.81  # reduced gravity
        self.ball_radius = 0.2

        # Define the maze layout (0: path, 1: wall)
        self.maze = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 1, 0, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1]
        ])

        # Define start and goal positions
        self.start_pos = np.array([1.0, 1.0])
        self.goal_pos = np.array([7.0, 7.0])

        # Action space: (tilt_x, tilt_y) each in range [-max_tilt_angle, max_tilt_angle]
        self.action_space = spaces.Box(
            low=-self.max_tilt_angle,
            high=self.max_tilt_angle,
            shape=(2,),
            dtype=np.float32
        )

        # Observation space: 128x128 grayscale image
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.maze.shape, dtype=np.uint8
        )

        # Initialize state
        self.state = None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize ball position and velocity
        self.state = np.array([
            self.start_pos[0],
            self.start_pos[1],
            0.0,  # initial velocity x
            0.0  # initial velocity y
        ], dtype=np.float32)

        # Calculate initial distance to goal in straight line
        self.prev_distance = np.linalg.norm(self.state[:2] - self.goal_pos)

        return self._get_observation(), {}

    def _get_observation(self):
        """ Get the current observation (image) of the environment. """
        obs = np.zeros(self.maze.shape, dtype=np.uint8)
        obs[self.maze == 1] = 255  # Walls
        x, y, _, _ = self.state

        obs[int(y), int(x)] = 128  # Ball position
        return obs

    def _check_collision(self, x, y):
        """
        Check if the ball at (x, y) (with radius self.ball_radius) collides with any wall.
        Walls are assumed to be centered on grid coordinates (where self.maze == 1)
        and have a square shape with side length self.wall_thickness.
        Only a local neighborhood around the ball is checked.
        """
        # Determine the ball's grid coordinate
        grid_x = int(x)
        grid_y = int(y)

        # Define the local neighborhood radius (in cells) to check.
        # Here, a radius of 1 means we check grid_x - 1 to grid_x + 1 (3 cells in x and y).
        neighborhood = 1

        # Determine the bounds of the neighborhood
        rows, cols = self.maze.shape
        min_row = max(0, grid_y - neighborhood)
        max_row = min(rows - 1, grid_y + neighborhood)
        min_col = max(0, grid_x - neighborhood)
        max_col = min(cols - 1, grid_x + neighborhood)

        # Check each cell in the neighborhood that is a wall (maze value == 1)
        for r in range(min_row, max_row + 1):
            for c in range(min_col, max_col + 1):
                if self.maze[r, c] == 1:
                    # Define the rectangle for this wall cell.
                    # The wall is centered at (c, r) with half-size = wall_thickness/2.
                    wall_left = c - self.wall_thickness / 2
                    wall_right = c + self.wall_thickness / 2
                    wall_top = r - self.wall_thickness / 2
                    wall_bottom = r + self.wall_thickness / 2

                    # Find the closest point on the rectangle to the ball's center.
                    nearest_x = np.clip(x, wall_left, wall_right)
                    nearest_y = np.clip(y, wall_top, wall_bottom)

                    # Compute the distance from the ball center to that nearest point.
                    dx = x - nearest_x
                    dy = y - nearest_y
                    if (dx * dx + dy * dy) < (self.ball_radius ** 2):
                        return True  # Collision detected

        return False  # No collision found

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """
        Render the current state of the environment.
        Supports both 'human' (renders to screen) and 'rgb_array' (returns image).
        """
        # Create a new figure
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot the maze
        ax.imshow(self.maze, cmap='binary', origin='upper')

        # Plot the ball
        x, y, _, _ = self.state
        circle = plt.Circle((x, y), self.ball_radius, color='blue')
        ax.add_artist(circle)

        # Plot the goal
        goal_circle = plt.Circle(self.goal_pos, self.ball_radius, color='green')
        ax.add_artist(goal_circle)

        # Set plot limits and aspect
        ax.set_xlim(0, self.maze_size[0])
        ax.set_ylim(0, self.maze_size[1])
        ax.set_aspect('equal')

        if self.render_mode == "rgb_array":
            # Render the plot as an RGB array
            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]  # Convert RGBA to RGB
            plt.close(fig)  # Close figure to prevent memory leaks
            return frame  # Return the image

        elif self.render_mode == "human":
            plt.show()  # Display the image
            return None  # No return value needed for human mode

        else:
            raise ValueError(f"Unsupported render mode: {self.render_mode}")

    def step(self, action):
        # Extract current state
        x, y, vx, vy = self.state
        tilt_x, tilt_y = action

        # Convert tilt angles to acceleration components
        ax = self.gravity * np.sin(np.radians(tilt_x))
        ay = self.gravity * np.sin(np.radians(tilt_y))

        # Apply friction
        friction_x = -self.friction * vx
        friction_y = -self.friction * vy

        # Update velocities
        vx += (ax + friction_x) * self.dt
        vy += (ay + friction_y) * self.dt

        # Update positions
        new_x = x + vx * self.dt
        new_y = y + vy * self.dt

        if self._check_collision(new_x, new_y):
            # Bounce back with some energy loss
            vx *= -0.5
            vy *= -0.5
            new_x = x
            new_y = y

        # Update state
        self.state = np.array([new_x, new_y, vx, vy], dtype=np.float32)

        # Calculate reward and check if done
        distance_to_goal = np.linalg.norm(self.state[:2] - self.goal_pos)

        reward = self.prev_distance - distance_to_goal

        self.prev_distance = distance_to_goal

        # Check if reached goal
        terminated = distance_to_goal < 0.1
        if terminated:
            reward = 100.0

        truncated = False

        obs = self._get_observation()

        return obs, reward, terminated, truncated, {}
