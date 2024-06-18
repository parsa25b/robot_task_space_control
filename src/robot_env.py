from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv

import numpy as np
from pathlib import Path
import mujoco

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

    def step(self, action: np.ndarray):
        """
        Perform a simulation step in the environment.

        Args:
            action: The action to be taken in the environment.

        """

        np.clip(
            action,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
            out=action,
        )
        self.do_simulation(action, self.frame_skip)

    def reset_model(self):
        """
        Resets the model to its initial state.
        """

        # Reset the position and control values with noise
        self.data.qpos[:] = self.model.key_qpos[0]
        self.data.ctrl[:] = self.model.key_ctrl[0]

        # Reset the variables and sample a new desired velocity
        self._step = 0

    @property
    def number_of_dofs(self):
        return self.model.nv

    @property
    def joint_range(self):
        return self.model.jnt_range

    def get_jacobian(self, site_id: int) -> np.ndarray:
        # Calculate Jacobian
        jacp = np.zeros((3, self.number_of_dofs))
        jacr = np.zeros((3, self.number_of_dofs))
        mujoco.mj_jacSite(
            self.model,
            self.data,
            jacp,
            jacr,
            site_id,
        )

        return np.vstack((jacp, jacr))

    def get_site_position(self, site_id: int) -> np.ndarray:
        return self.data.site(site_id).xpos

    def get_site_quaternion(self, site_id: int) -> np.ndarray:
        site_quat = np.zeros(4)
        mujoco.mju_mat2Quat(site_quat, self.data.site(site_id).xmat)
        return site_quat

    def get_qpos(self) -> np.ndarray:
        return self.data.qpos

    def set_ctrl(self, ctrl: np.ndarray):
        self.data.ctrl = ctrl

    def set_qpos(self, qpos: np.ndarray):
        self.data.qpos = qpos
