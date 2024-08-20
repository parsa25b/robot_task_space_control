from os import path
from pathlib import Path
import mujoco
import mujoco_viewer
import numpy as np

DEFAULT_CAMERA_CONFIG = {
    "azimuth": 100.0,
    "distance": 2.0,
    "elevation": -25.0,
    "lookat": np.array([0.0, 0.0, 0.2]),
    "fixedcamid": -1,
    "trackbodyid": -1,
    "type": 0,
}


class RobotEnv:
    def __init__(
        self, model_path: Path, unlocked_joint_name: list, render: bool = True
    ):
        model_path = model_path.absolute().as_posix()

        if model_path.startswith(".") or model_path.startswith("/"):
            self.fullpath = model_path
        elif model_path.startswith("~"):
            self.fullpath = path.expanduser(model_path)
        else:
            self.fullpath = path.join(path.dirname(__file__), "assets", model_path)
        if not path.exists(self.fullpath):
            raise OSError(f"File {self.fullpath} does not exist")

        if render:
            self.render_mode = "window"
        else:
            self.render_mode = "offscreen"
        self._initialize_simulation()
        self._unlocked_joint_idx = [
            self.joint_name_to_id(joint) for joint in unlocked_joint_name
        ]

    def _initialize_simulation(self):
        self.model = mujoco.MjModel.from_xml_path(self.fullpath)
        self.data = mujoco.MjData(self.model)
        # initialize the viewer
        self.viewer = mujoco_viewer.MujocoViewer(
            self.model, self.data, mode=self.render_mode
        )
        self._reset_simulation()

    def _reset_simulation(self):
        mujoco.mj_resetData(self.model, self.data)

    def render(self):
        if self.render_mode == "window":
            self.viewer.render()
        # elif self.render_mode == 'offscreen':
        #     self.viewer.read_pixels()

    def step(self):
        """
        Perform a simulation step in the environment.

        """

        np.clip(
            self.data.qpos,
            self.model.jnt_range[:, 0],
            self.model.jnt_range[:, 1],
            out=self.data.qpos,
        )
        mujoco.mj_step(self.model, self.data)

    def forward_dynamics(self):
        """
        Applies forward dynamics to the robot's model and data.
        """
        np.clip(
            self.data.qpos,
            self.model.jnt_range[:, 0],
            self.model.jnt_range[:, 1],
            out=self.data.qpos,
        )
        mujoco.mj_forward(self.model, self.data)

    def reset_model(self):
        """
        Resets the model to its initial state.
        """

        # Reset the position and control values with noise
        self.data.qpos[:] = self.model.key_qpos[0]
        self.data.ctrl[:] = self.model.key_ctrl[0]

    def get_jacobian(self, frame_name: str) -> np.ndarray:
        frame_id = self.body_name_to_id(frame_name)
        # Calculate Jacobian
        jacp = np.zeros((3, self.number_of_dofs))
        jacr = np.zeros((3, self.number_of_dofs))
        mujoco.mj_jacBody(
            self.model,
            self.data,
            jacp,
            jacr,
            frame_id,
        )

        return np.vstack((jacp, jacr))[:, self.unlocked_joint_idx]

    def get_site_jacobian(self, site_name: str) -> np.ndarray:
        site_id = self.site_name_to_id(site_name)
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

    def get_body_position(self, body_name: str) -> np.ndarray:
        body_id = self.body_name_to_id(body_name)
        return self.data.body(body_id).xpos

    def get_site_position(self, site_name: str) -> np.ndarray:
        site_id = self.site_name_to_id(site_name)
        return self.data.site(site_id).xpos

    def get_body_orientation(self, body_name: str) -> np.ndarray:
        body_id = self.body_name_to_id(body_name)
        return self.data.body(body_id).xquat

    def get_site_quaternion(self, site_name: str) -> np.ndarray:
        site_quat = np.zeros(4)
        site_id = self.site_name_to_id(site_name)
        mujoco.mju_mat2Quat(site_quat, self.data.site(site_id).xmat)
        return site_quat

    def get_full_qpos(self) -> np.ndarray:
        return self.data.qpos

    def get_qpos(self) -> np.ndarray:
        return self.data.qpos[self.unlocked_joint_idx]

    def get_bias_forces(self) -> np.ndarray:
        return self.data.qfrc_bias[self.unlocked_joint_idx]

    def set_ctrl(self, ctrl: np.ndarray):
        self.data.ctrl = ctrl

    def set_qpos(self, qpos: np.ndarray):
        self.data.qpos[self.unlocked_joint_idx] = qpos

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray):
        q_full = self.data.qpos[:]
        qdot_full = self.data.qvel[:]
        q_full[self.unlocked_joint_idx] = qpos
        qdot_full[self.unlocked_joint_idx] = qvel
        self.data.qpos[:] = q_full
        self.data.qvel[:] = qdot_full

    def set_full_state(self, qpos: np.ndarray):
        self.data.qpos = qpos

    def joint_name_to_id(self, joint_name: str):
        """
        Returns the joint ID given the joint name
        """
        return mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT.value, joint_name
        )

    def body_name_to_id(self, body_name: str):
        """
        Returns the body ID given the body name
        """
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY.value, body_name)

    def site_name_to_id(self, site_name: str):
        """
        Returns the site ID given the site name
        """
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE.value, site_name)

    def set_timestep(self, timestep: float):
        self.model.opt.timestep = timestep

    def add_marker(
        self,
        xpos: np.ndarray,
        label: str,
        marker_size: float = 0.05,
        marker_color: np.ndarray = np.array([1.0, 0.0, 0.0, 1.0]),
    ):
        """
        Add a marker to the simulation
        """
        if xpos.shape != (3,):
            raise ValueError("xpos must be a 3D vector")
        self.viewer.add_marker(
            pos=xpos,
            label=label,
            type=2,
            size=marker_size * np.ones(3),
            rgba=marker_color,
        )

    def close_renderer(self):
        self.viewer.close()

    @property
    def number_of_dofs(self):
        return self.model.nv

    @property
    def number_of_unlocked_joints(self):
        return len(self.unlocked_joint_idx)

    @property
    def joint_range(self):
        return self.model.jnt_range[self.unlocked_joint_idx]

    @property
    def joint_names(self):
        joint_names = []
        for i in range(self.model.nv):
            joint_names.append(self.data.joint(i).name)

        return joint_names

    @property
    def unlocked_joint_names(self):
        joint_names = []
        for i in self.unlocked_joint_idx:
            joint_names.append(self.data.joint(i).name)

        return joint_names

    @property
    def unlocked_joint_idx(self):
        return self._unlocked_joint_idx

    @property
    def body_names(self):
        body_names = []
        for i in range(self.model.nbody):
            body_names.append(self.data.body(i).name)

        return body_names

    @property
    def timestep(self):
        return self.model.opt.timestep
