import mujoco
import numpy as np
from robot_env import RobotEnv


# Levenberg-Marquardt method
class LevenbegMarquardtIK:
    """
    Levenberg-Marquardt Inverse Kinematics solver for a robot arm.

    Args:
        env (RobotEnv): The robot environment.
        step_size (float): The step size for each iteration.
        damping (float): The damping factor for the Levenberg-Marquardt algorithm.
    """

    def __init__(self, env: RobotEnv):
        self.env = env
        self.step_size = 0.5
        self.damping = 0.1

    def check_joint_limits(self, q):
        """Check if the joints are within their limits."""
        for i in range(len(q)):
            q[i] = max(
                self.env.joint_range[i][0], min(q[i], self.env.joint_range[i][1])
            )

    def calculate(
        self,
        goal: np.ndarray,
        frame_name: str,
        type: str = "body",
        number_of_iterations=3,
    ):
        """
        Calculate the desired joint angles for a given goal full pose.

        Args:
            goal (numpy.ndarray): The desired full pose.
            site_name (str): The name of the site to control.
        """

        # Calculate trnaslational and rotational error
        error = np.zeros(6)
        error_pos = error[:3]
        error_ori = error[3:]
        site_quat_conj = np.zeros(4)
        error_quat = np.zeros(4)
        site_quat = np.zeros(4)
        if type == "body":
            current_pos = self.env.get_body_position(frame_name)
            site_quat_conj = self.env.get_body_orientation(frame_name)
            site_quat_conj[0] = -site_quat_conj[0]
        elif type == "site":
            current_pos = self.env.get_site_position(frame_name)
            site_quat = self.env.get_site_quaternion(frame_name)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
        error_pos = np.subtract(goal[:3], current_pos)
        mujoco.mju_mulQuat(error_quat, goal[3:], site_quat_conj)
        mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)

        error = np.concatenate((error_pos, error_ori))

        iteration = 0
        qpos_before_ik = self.env.get_qpos().copy()
        while iteration < number_of_iterations:
            # Calculate Jacobian
            if type == "body":
                jac = self.env.get_jacobian(frame_name)
            elif type == "site":
                jac = self.env.get_site_jacobian(frame_name)

            # jac = jac[:3]
            # error = error[:3]

            # Calculate delta of joint q
            n = jac.shape[1]
            I = np.identity(n)
            product = jac.T @ jac + self.damping * I

            if np.isclose(np.linalg.det(product), 0):
                j_inv = np.linalg.pinv(product) @ jac.T
            else:
                j_inv = np.linalg.inv(product) @ jac.T
            delta_q = j_inv @ error

            # Compute next step
            qpos = self.env.get_qpos() + self.step_size * delta_q

            # Check joint limits
            self.check_joint_limits(qpos)

            # Set joint positions
            self.env.set_qpos(qpos=qpos)

            # Compute forward kinematics
            mujoco.mj_forward(self.env.model, self.env.data)

            # Calculate new error
            if type == "body":
                current_pos = self.env.get_body_position(frame_name)
                site_quat_conj = self.env.get_body_orientation(frame_name)
                site_quat_conj[0] = -site_quat_conj[0]
            elif type == "site":
                current_pos = self.env.get_site_position(frame_name)
                site_quat = self.env.get_site_quaternion(frame_name)
                mujoco.mju_negQuat(site_quat_conj, site_quat)

            error_pos = np.subtract(goal[:3], current_pos)
            mujoco.mju_mulQuat(error_quat, goal[3:], site_quat_conj)
            mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)
            error = np.concatenate((error_pos, error_ori))

            iteration += 1

        self.env.set_qpos(qpos=qpos_before_ik)
        return qpos
