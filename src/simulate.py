import argparse
import time
import mujoco_viewer
from pathlib import Path
# from ik_levenberg_marquardt import LevenbegMarquardtIK
from ik_gradient_descent import GradientDescentIK
from tqdm import tqdm
import mujoco
import numpy as np
from robot_env import RobotEnv


def circle(t: float, r: float, h: float, k: float, f: float) -> np.ndarray:
    """Return the (x, y) coordinates of a circle with radius r centered at (h, k)
    as a function of time t and frequency f."""
    x = r * np.cos(2 * np.pi * f * t) + h
    y = r * np.sin(2 * np.pi * f * t) + k
    z = 0.5
    return np.array([x, y, z])


def simulate(args):
    """
    Simulates the robot control loop.

    Args:
        args (object): The arguments object containing the simulation parameters.
    """

    # Environment setup
    env = RobotEnv(model_path= Path("universal_robots_ur5e/ur5e.xml"),
        render_mode="rgb_array",
    )
    inter_frame_sleep = 0.016
    # model = PPO.load(path=model_path, env=env, verbose=1)
    num_episodes = 1
    total_reward = 0
    total_length = 0

    # Inverse Kinematics setup
    jacp = np.zeros((3, env.model.nv))  # translation jacobian
    jacr = np.zeros((3, env.model.nv))  # rotational jacobian
    step_size = 0.5
    tol = 0.01
    alpha = 2.0
    damping = 0.15
    # ik = LevenbegMarquardtIK(
    #     env.model, env.data, step_size, tol, jacp, jacr, damping, maxiter=50
    # )
    ik = GradientDescentIK(env.model, env.data, step_size, tol, alpha, jacp, jacr)

    # body_id = env.model.body("wrist_3_link").id  # "End-effector we wish to control.
    # End-effector we wish to control.
    site_name = "end_effector"
    site_id = env.model.site(site_name).id
    mocap_id = env.model.body("target").mocapid[0]

    for _ in tqdm(range(num_episodes)):
        obs, _ = env.reset()

        # Initialize the mujoco viewer #TODO add this to the UR5eMujocoEnv instead of using mujoco_py
        viewer = mujoco_viewer.MujocoViewer(env.model, env.data)
        # Reset the simulation
        mujoco.mj_resetDataKeyframe(env.model, env.data, 0)

        ep_len = 0
        ep_reward = 0
        initialize_simulation_loop = True
        env.data.mocap_pos[mocap_id] = [0.49199893, 0.13399819, 0.58800037]

        while True:
            # ee_reference_pose = circle(env.data.time, 0.1, 0.3, 0.0, 0.5)  # ENABLE to test circle.
            ee_reference_pose = np.concatenate(
                (env.data.mocap_pos[mocap_id], env.data.mocap_quat[mocap_id])
            )
            # ee_reference_pose = [0.49199893, 0.13399819, 0.58800037]
            # t = env.data.time
            # point_in_body_frame = [0, 0, -0.1]
            # viewer.add_marker(
            #     pos=env.data.body(site_id).xpos + point_in_body_frame,
            #     label=str(t),
            #     size=[0.05, 0.05, 0.05],
            # )

            # action, _ = model.predict(obs, deterministic=True)
            if initialize_simulation_loop:
                init_q = env.data.qpos.copy()
                initialize_simulation_loop = False
            else:
                init_q = q_previous_ik

            # Inverse Kinematics calculations
            ik.calculate(ee_reference_pose, init_q, site_id)  # calculate the qpos
            q_previous_ik = env.data.qpos.copy()
            action = env.data.qpos

            # action = env.model.key_qpos[0]

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1

            current_pose = env.data.body(site_id).xpos
            # print(f"Current pose: {current_pose}")
            print(f"ee_reference_pose: {ee_reference_pose}")
            print(f"data.site_xpos={env.data.site_xpos}")
            viewer.render()
            # Slow down the rendering
            time.sleep(inter_frame_sleep)
            if terminated or truncated:
                print(f"{ep_len=}  {ep_reward=}")
                break

        total_length += ep_len
        total_reward += ep_reward
        viewer.close()

    print(
        f"Avg episode reward: {total_reward / num_episodes}, avg episode length: {total_length / num_episodes}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",type=str,default= "universal_robots_ur5e/ur5e.xml",help="Path to the robot model")
    args = parser.parse_args()

    simulate(args)
