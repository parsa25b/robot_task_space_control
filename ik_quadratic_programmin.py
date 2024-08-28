import mujoco
import numpy as np
import mujoco

# Gradient Descent method
class QuadraticProgrammingIK:
    """
    A class that implements the Quadratic Programming Inverse Kinematics algorithm.

    Parameters:
    - model: The Mujoco model object.
    - data: The Mujoco data object.
    """

    def __init__(self, model, data, kp, tol, alpha, jacp, jacr):
        self.model = model
        self.data = data
        self.kp = kp
        self.tol = tol
        self.alpha = alpha
        self.jacp = jacp
        self.jacr = jacr
        self.initialize_calculate = False

    def check_joint_limits(self, q):
        """Check if the joints are within their limits."""
        for i in range(len(q)):
            q[i] = max(
                self.model.jnt_range[i][0], min(q[i], self.model.jnt_range[i][1])
            )

    # Gradient Descent pseudocode implementation
    def calculate(self, goal, goal_dot, init_q, body_id, dt):
        """
        Calculate the desired joint angles for a given cartesian goal position.

        Args:
            goal (numpy.ndarray): The desired cartesian goal position.
            init_q (numpy.ndarray): The initial joint angles.
            body_id (int): The ID of the body to control.
        """

        self.data.qpos = init_q
        mujoco.mj_forward(self.model, self.data)
        current_pose = self.data.body(body_id).xpos
        error = np.subtract(goal, current_pose)

        while np.linalg.norm(error) >= self.tol:
            # Calculate Jacobian
            mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, goal, body_id)

            M = self.jacp
            P = np.dot(A.T, A)
            pre_b`` = self.kp * (goal - current_pose) + goal_dot
            g = -np.dot(A.T, b)

            A = np.vstack((self.jacp * dt, -self.jacp * dt))
            # Calculate gradient
            grad = self.alpha * self.jacp.T @ error
            # Compute next step
            self.data.qpos += self.step_size * grad
            # Check joint limits
            self.check_joint_limits(self.data.qpos)
            # Compute forward kinematics
            mujoco.mj_forward(self.model, self.data)
            # Calculate new error
            error = np.subtract(goal, self.data.body(body_id).xpos)
