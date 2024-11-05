import numpy as np
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation as R
from mujoco_robot_control.robot_env import RobotEnv


# Levenberg-Marquardt method
class QuadraticProgrammingIK:
    """
    Quadratic Programming Inverse Kinematics solver.

    Args:
        env (RobotEnv): The robot environment.
    """

    def __init__(self, env: RobotEnv):
        self.env = env
        self.penalize_joint_velocity = 1e-1
        self.penalize_delta_joint_velocity = 1e-3
        self.penalize_joint_position = 1e-1
        self.previous_ik_qpos = np.zeros(self.env.number_of_unlocked_joints)
        self.xd_previous = None
        self.delta_q_previous = 0

    def calculate(
        self,
        goal: np.ndarray,
        frame_name: str,
        weight: np.ndarray = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        frame_type: str = "body",
        number_of_iterations=1,
        only_position=True,
    ):
        """
        Calculate the desired joint angles for a given goal full pose.

        Args:
            goal (numpy.ndarray): The desired full pose.
            frame_name (str): The name of the frame to control.
            weight (numpy.ndarray): The weight for each dimension of the error.
            frame_type (str): The type of frame to control (either 'body' or 'site').
            number_of_iterations (int): The number of iterations to run the IK solver.
            only_position (bool): Whether to only control the position or both position and orientation
        """

        xd = np.zeros(6)
        xd[:3] = goal[:3]
        quaternion = [goal[6], goal[3], goal[4], goal[5]]
        xd[3:] = R.from_quat(quaternion).as_rotvec()
        if self.xd_previous is None:
            self.xd_previous = xd.copy()
        xddot = (xd - self.xd_previous) / self.env.timestep

        qpos_before_ik = self.env.get_qpos().copy()
        for _ in range(number_of_iterations):
            error = self._caclculate_error(frame_name, goal, frame_type)
            # Calculate Jacobian
            if frame_type == "body":
                jac = self.env.get_jacobian(frame_name)
            elif frame_type == "site":
                jac = self.env.get_site_jacobian(frame_name)

            if only_position:
                jac = jac[:3]
                error = error[:3]
                xddot = xddot[:3]
                weight = weight[:3]
                weight = np.diag(weight)
            else:
                weight = np.diag(weight)

            error = np.clip(error, -0.1, 0.1)
            A = weight @ jac
            b = weight @ error * 10
            P = A.T @ A
            P += self.penalize_joint_velocity * np.eye(P.shape[0])
            # regularization term on joint velocities
            P += self.penalize_delta_joint_velocity * np.eye(P.shape[0])
            # regularization term on joint positions
            P += self.penalize_joint_position * np.eye(P.shape[0])
            P = sparse.csc_matrix(P)
            q = -A.T @ b
            q -= self.penalize_delta_joint_velocity * self.delta_q_previous
            q += self.penalize_joint_position * self.previous_ik_qpos
            l = (self.env.joint_range[:, 0] - self.previous_ik_qpos) / self.env.timestep
            u = (self.env.joint_range[:, 1] - self.previous_ik_qpos) / self.env.timestep
            C = sparse.eye(len(q), format="csc")
            # Solve QP
            osqp_solver = osqp.OSQP()
            osqp_solver.setup(P=P, q=q, l=l, u=u, A=C, verbose=False)
            result = osqp_solver.solve()

            delta_q = result.x
            if result.info.status != "solved":
                return self.previous_ik_qpos, self.delta_q_previous / self.env.timestep

            # Compute next step
            qpos = self.env.get_qpos() + delta_q * self.env.timestep
            self.env.set_qpos(qpos=qpos)
            self.env.forward_dynamics()

        self.env.set_qpos(qpos=qpos_before_ik)
        self.env.forward_dynamics()

        self.delta_q_previous = delta_q
        self.xd_previous = xd
        self.previous_ik_qpos = qpos

        return qpos, delta_q / self.env.timestep

    def _caclculate_error(self, frame_name, goal, frame_type):
        """
        Calculate the error between the current pose and the goal pose.
        """

        if frame_type == "body":
            current_pos = self.env.get_body_position(frame_name)
            site_quat = self.env.get_body_orientation(frame_name)

        elif frame_type == "site":
            current_pos = self.env.get_site_position(frame_name)
            site_quat = self.env.get_site_quaternion(frame_name)
            site_quat_conj = self._neg_quat(site_quat)

        else:
            raise ValueError("Invalid type. Please use 'body' or 'site'.")

        site_quat_conj = self._neg_quat(site_quat)
        error_pos = goal[:3] - current_pos
        error_quat = self._mul_quat(goal[3:], site_quat_conj)
        error_ori = self._quat_to_velocity(error_quat)
        error = np.concatenate((error_pos, error_ori))

        return error

    @staticmethod
    def _quat_to_velocity(error_quat):
        """
        Converts a quaternion difference into an angular velocity vector.

        Parameters:
        - error_quat: The quaternion representing the orientation error (array-like of length 4).

        Returns:
        - error_ori: The angular velocity vector corresponding to the quaternion difference (array-like of length 3).
        """
        # Ensure the quaternion is normalized
        error_quat = np.array(error_quat)
        if np.linalg.norm(error_quat) != 0:
            error_quat = error_quat / np.linalg.norm(error_quat)

        # Extract the vector part (x, y, z) and the scalar part (w) of the quaternion
        q_vec = error_quat[1:]  # x, y, z
        q_w = error_quat[0]  # w

        # Calculate the angular velocity
        if q_w != 0:
            error_ori = 2.0 * q_vec / q_w
        else:
            error_ori = 2.0 * q_vec  # Avoid division by zero, assumes w close to zero

        return error_ori

    @staticmethod
    def _mul_quat(q1, q2):
        """
        Multiplies two quaternions.

        Parameters:
        - q1: The first quaternion (array-like of length 4).
        - q2: The second quaternion (array-like of length 4).

        Returns:
        - result_quat: The resulting quaternion from the multiplication (array-like of length 4).
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return np.array([w, x, y, z])

    @staticmethod
    def _neg_quat(quat):
        """
        Negates a quaternion (i.e., computes the conjugate).

        Parameters:
        - quat: The input quaternion (array-like of length 4).

        Returns:
        - negated_quat: The negated quaternion (array-like of length 4).
        """
        # Keep the scalar part (w) the same, negate the vector part (x, y, z)
        w, x, y, z = quat
        negated_quat = np.array([w, -x, -y, -z])

        return negated_quat
