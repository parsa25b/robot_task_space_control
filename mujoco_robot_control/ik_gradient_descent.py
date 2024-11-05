import mujoco
import numpy as np
from robot_env import RobotEnv


# Gradient Descent method
class GradientDescentIK:
    """
    A class that implements the Gradient Descent Inverse Kinematics algorithm.

    Parameters:
    - env: The robot model environment object.
    """

    def __init__(self, env: RobotEnv):
        self.env = env
        self.step_size = 0.5
        self.alpha = 1.0
        self.tol = 0.01

    def check_joint_limits(self, q):
        """Check if the joints are within their limits."""
        for i in range(len(q)):
            q[i] = max(
                self.env.joint_range[i][0], min(q[i], self.env.joint_range[i][1])
            )

    def calculate(self, goal, site_id, maxiter=5):
        """
        Calculate the desired joint angles for a given goal full pose.

        Args:
            goal (numpy.ndarray): The desired full pose.
            site_id (int): The ID of the site to control.
        """
        # self.data.qpos = init_q
        # mujoco.mj_forward(self.model, self.data)

        # Calculate trnaslational and rotational error
        error = np.zeros(6)
        error_pos = error[:3]
        error_ori = error[3:]
        site_quat_conj = np.zeros(4)
        error_quat = np.zeros(4)
        site_quat = np.zeros(4)

        current_pos = self.env.get_site_position(site_id)
        error_pos = np.subtract(goal[:3], current_pos)

        site_quat = self.env.get_site_quaternion(site_id=site_id)
        mujoco.mju_negQuat(site_quat_conj, site_quat)
        mujoco.mju_mulQuat(error_quat, goal[3:], site_quat_conj)
        mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)

        # error = np.concatenate((error_pos, error_ori))

        iter = 0
        qpos_before_ik = self.env.get_qpos().copy()
        while iter <= maxiter:
            # Calculate Jacobian
            jac = self.env.get_jacobian(site_id=site_id)

            # Calculate gradient
            grad = self.alpha * jac.T @ error

            # Compute next step
            qpos = self.env.get_qpos() + self.step_size * grad

            # Check joint limits
            self.check_joint_limits(qpos)

            # Set joint positions
            self.env.set_qpos(qpos=qpos)

            # Compute forward kinematics
            mujoco.mj_forward(self.env.model, self.env.data)

            # Calculate new error
            current_pos = self.env.get_site_position(site_id)
            error_pos = np.subtract(goal[:3], current_pos)
            site_quat = self.env.get_site_quaternion(site_id=site_id)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, goal[3:], site_quat_conj)
            mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)
            error = np.concatenate((error_pos, error_ori))

            iter += 1

        self.env.set_qpos(qpos=qpos_before_ik)
        return qpos
