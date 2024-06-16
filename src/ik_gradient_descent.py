import mujoco
import numpy as np


# Gradient Descent method
class GradientDescentIK:
    """
    A class that implements the Gradient Descent Inverse Kinematics algorithm.

    Parameters:
    - model: The Mujoco model object.
    - data: The Mujoco data object.
    - step_size: The step size for each iteration of the algorithm.
    - tol: The tolerance value for convergence.
    - alpha: The scaling factor for the gradient.
    - jacp: The Jacobian matrix for position.
    - jacr: The Jacobian matrix for rotation.
    """

    def __init__(self, model, data, step_size, tol, alpha, number_of_dofs, maxiter=1):
        self.model = model
        self.data = data
        self.step_size = step_size
        self.tol = tol
        self.alpha = alpha
        self.jacp = np.zeros((3, number_of_dofs))  # translation jacobian
        self.jacr = np.zeros((3, number_of_dofs))  # rotational jacobian
        self.maxiter = maxiter

    def check_joint_limits(self, q):
        """Check if the joints are within their limits."""
        for i in range(len(q)):
            q[i] = max(
                self.model.jnt_range[i][0], min(q[i], self.model.jnt_range[i][1])
            )

    # Gradient Descent pseudocode implementation
    def calculate(self, goal, init_q, site_id):
        """
        Calculate the desired joint angles for a given cartesian goal position.

        Args:
            goal (numpy.ndarray): The desired cartesian goal position.
            init_q (numpy.ndarray): The initial joint angles.
            body_id (int): The ID of the body to control.
        """
        self.data.qpos = init_q
        mujoco.mj_forward(self.model, self.data)
        current_pos = self.data.site(site_id).xpos

        # error = np.zeros(6)
        # error_pos = error[:3]
        # error_ori = error[3:]
        # site_quat_conj = np.zeros(4)
        # error_quat = np.zeros(4)
        # site_quat = np.zeros(4)

        # error_pos = np.subtract(goal[:3], current_pos)
        # mujoco.mju_mat2Quat(site_quat, self.data.site(site_id).xmat)
        # mujoco.mju_negQuat(site_quat_conj, site_quat)
        # mujoco.mju_mulQuat(error_quat, goal[3:], site_quat_conj)
        # mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)
        # error = np.concatenate((error_pos, error_ori))

        error = np.subtract(goal[:3], current_pos)
        iter = 0
        while np.linalg.norm(error) >= self.tol and iter <= self.maxiter:
            # Calculate Jacobian
            mujoco.mj_jacSite(
                self.model,
                self.data,
                self.jacp,
                self.jacr,
                site_id,
            )

            # jac = np.concatenate((self.jacp[:, 3:], self.jacr[:, 3:]))
            jac = self.jacp

            # mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, site_id)
            # Calculate gradient
            grad = self.alpha * jac.T @ error
            # Compute next step
            self.data.qpos += self.step_size * grad
            # Check joint limits
            self.check_joint_limits(self.data.qpos)
            # Compute forward kinematics
            mujoco.mj_forward(self.model, self.data)
            current_pose = self.data.site(site_id).xpos
            # Calculate new error
            error_pos = np.subtract(goal[:3], current_pose)
            # mujoco.mju_mat2Quat(site_quat, self.data.site(site_id).xmat)
            # mujoco.mju_negQuat(site_quat_conj, site_quat)
            # mujoco.mju_mulQuat(error_quat, goal[3:], site_quat_conj)
            # mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)
            # error = np.concatenate((error_pos, error_ori))
            error = np.subtract(goal[:3], current_pos)
            iter += 1
