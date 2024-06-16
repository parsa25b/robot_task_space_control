from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
import mujoco

import numpy as np
from pathlib import Path

DEFAULT_CAMERA_CONFIG = {
    "azimuth": 100.0,
    "distance": 2.0,
    "elevation": -25.0,
    "lookat": np.array([0.0, 0.0, 0.2]),
    "fixedcamid": -1,
    "trackbodyid": -1,
    "type": 0,
}


class RobotEnv(MujocoEnv):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self, model_path: Path, **kwargs):

        MujocoEnv.__init__(
            self,
            model_path=model_path.absolute().as_posix(),
            frame_skip=10,  # Perform an action every 10 frames (dt(=0.002) * 10 = 0.02 seconds -> 50hz action rate)
            observation_space=None,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        # Update metadata to include the render FPS
        self.metadata["render_fps"] = 60

        self._last_render_time = -1.0
        self._max_episode_time_sec = 60.0
        self._step = 0

        self._clip_obs_threshold = 100.0
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self._get_obs().shape, dtype=np.float64
        )

        dof_position_limit_multiplier = 0.9  # The % of the range that is not penalized
        ctrl_range_offset = (
            0.5
            * (1 - dof_position_limit_multiplier)
            * (
                self.model.actuator_ctrlrange[:, 1]
                - self.model.actuator_ctrlrange[:, 0]
            )
        )
        # First value is the root joint, so we ignore it
        self._soft_joint_range = np.copy(self.model.actuator_ctrlrange)
        self._soft_joint_range[:, 0] += ctrl_range_offset
        self._soft_joint_range[:, 1] -= ctrl_range_offset
        self._reset_noise_scale = 0.1

    def step(self, action):
        """
        Perform a simulation step in the environment.

        Args:
            action: The action to be taken in the environment.

        Returns:
            observation: The current observation of the environment.
            reward: The reward obtained from the environment.
            terminated: A boolean indicating whether the episode is terminated.
            truncated: A boolean indicating whether the episode is truncated.
            info: Additional information about the environment.
        """

        self._step += 1
        np.clip(
            action,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
            out=action,
        )
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        reward, reward_info = self._get_reward(action)
        # TODO consider termination conditions
        terminated = not self.is_healthy
        truncated = self._step >= (self._max_episode_time_sec / self.dt)
        info = {
            # "end-effector": self.data.get_body_xpos("end-effector"),
        }

        if self.render_mode == "human":
            # self.render()
            self._last_render_time = self.data.time

        return observation, reward, terminated, truncated, info

    @property
    def is_healthy(self):
        """
        Checks the health status of the environment.

        Returns:
            bool: True if the environment is healthy, False otherwise.
        """
        state = self.state_vector()
        # TODO consider other health checks

        return True

    def _get_reward(self, action):
        """
        Calculate the reward based on the given action.

        Args:
            action: The action taken by the agent.

        Returns:
            reward: The calculated reward value.
            reward_info: Additional information about the reward.
        """

        reward = 0.0
        reward_info = {}
        return reward, reward_info

    def _get_obs(self):
        """
        Get the current observation of the environment.

        Returns:
            curr_obs: Current observation of the environment.
        """

        position_dofs = np.array([])
        velocity_dofs = np.array([])
        for i in range(self.data.qpos.size):
            position_dofs = np.append(
                position_dofs, self.data.qpos[i] - self.model.key_qpos[0, i]
            )

        for i in range(self.data.qvel.size):
            velocity_dofs = np.append(velocity_dofs, self.data.qvel[i])

        curr_obs = np.concatenate((position_dofs, velocity_dofs)).clip(
            -self._clip_obs_threshold, self._clip_obs_threshold
        )

        return curr_obs

    def reset_model(self):
        """
        Resets the model to its initial state.

        Returns:
            observation: The initial observation of the environment.
        """

        # Reset the position and control values with noise
        self.data.qpos[:] = self.model.key_qpos[0] + self.np_random.uniform(
            low=-self._reset_noise_scale,
            high=self._reset_noise_scale,
            size=self.model.nq,
        )
        self.data.ctrl[:] = self.model.key_ctrl[
            0
        ] + self._reset_noise_scale * self.np_random.standard_normal(
            *self.data.ctrl.shape
        )

        # Reset the variables and sample a new desired velocity
        self._step = 0
        self._last_render_time = -1.0

        observation = self._get_obs()

        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
        }
