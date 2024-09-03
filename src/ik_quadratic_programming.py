import numpy as np
import osqp
from robot_env import RobotEnv
from scipy import sparse
from scipy.spatial.transform import Rotation as R


# Levenberg-Marquardt method
class QuadraticProgrammingIK:
    """
    Quadratic Programming Inverse Kinematics solver.

    Args:
    """

    def __init__(self, env: RobotEnv):
        self.env = env
        self.step_size = 1
        self.xd_previous = None

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
        weight: np.ndarray = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        type: str = "body",
        number_of_iterations=1,
        only_position=True,
    ):
        """
        Calculate the desired joint angles for a given goal full pose.

        Args:
            goal (numpy.ndarray): The desired full pose.
            site_name (str): The name of the site to control.
        """
        xd = np.zeros(6)
        xd[:3] = goal[:3]
        quaternion = [goal[6], goal[3], goal[4], goal[5]]
        xd[3:] = R.from_quat(quaternion).as_rotvec()
        if self.xd_previous is None:
            self.xd_previous = xd

        xddot = (xd - self.xd_previous) / self.env.timestep

        weight = np.diag(weight)
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
            site_quat_conj = self.neg_quat(site_quat)
        else:
            raise ValueError("Invalid type. Please use 'body' or 'site'.")

        error_pos = np.subtract(goal[:3], current_pos)
        error_quat = self.mul_quat(goal[3:], site_quat_conj)
        error_ori = self.quat_to_velocity(error_quat)
        error = np.concatenate((error_pos, error_ori))

        iteration = 0
        qpos_before_ik = self.env.get_qpos().copy()
        while iteration < number_of_iterations:
            # Calculate Jacobian
            if type == "body":
                jac = self.env.get_jacobian(frame_name)
            elif type == "site":
                jac = self.env.get_site_jacobian(frame_name)

            if only_position:
                jac = jac[:3]
                error = error[:3]
                xddot = xddot[:3]

            A = jac
            b = xddot + error
            P = np.dot(A.T, A)
            P += 1e-3 * np.eye(P.shape[0])
            P = sparse.csc_matrix(P)
            q = -np.dot(A.T, b)
            # Solve QP
            osqp_solver = osqp.OSQP()
            osqp_solver.setup(P=P, q=q, verbose=False)
            result = osqp_solver.solve()
            delta_q = result.x
            if result.info.status != "solved":
                raise ValueError("OSQP did not find a solution.")

            # Compute next step
            qpos = self.env.get_qpos() + self.step_size * delta_q * self.env.timestep

            self.check_joint_limits(qpos)
            self.env.set_qpos(qpos=qpos)

            self.env.forward_dynamics()

            if type == "body":
                current_pos = self.env.get_body_position(frame_name)
                site_quat_conj = self.env.get_body_orientation(frame_name)
                site_quat_conj[0] = -site_quat_conj[0]
            elif type == "site":
                current_pos = self.env.get_site_position(frame_name)
                site_quat = self.env.get_site_quaternion(frame_name)
                site_quat_conj = self.neg_quat(site_quat)
            else:
                raise ValueError("Invalid type. Please use 'body' or 'site'.")

            error_pos = np.subtract(goal[:3], current_pos)
            error_quat = self.mul_quat(goal[3:], site_quat_conj)
            error_ori = self.quat_to_velocity(error_quat)
            error = np.concatenate((error_pos, error_ori))

            iteration += 1

        self.env.set_qpos(qpos=qpos_before_ik)
        self.xd_previous = xd
        return qpos

    @staticmethod
    def quat_to_velocity(error_quat):
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
    def mul_quat(q1, q2):
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
    def neg_quat(quat):
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
