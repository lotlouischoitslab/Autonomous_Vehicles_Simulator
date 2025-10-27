"""
Self-Driving Car Simulator (Gymnasium Env) + SAC training script
----------------------------------------------------------------
- Bicycle model dynamics (continuous control: throttle/accel, steering rate)
- Single lane-follow task along a closed loop of waypoints
- Static circular obstacles
- Dense reward for forward progress, lane keeping, smoothness; large penalty for off-road/collision
- Minimal pygame renderer
- Trains with Stable-Baselines3 SAC (continuous actions)

Usage
-----
pip install gymnasium pygame stable-baselines3[extra] torch numpy scipy
python self_driving_rl_gym_env.py --train  # trains and saves best model
python self_driving_rl_gym_env.py --eval   # runs a short evaluation with rendering

Notes
-----
- This file is self-contained. No external assets required.
- Designed for clarity and extensibility (waypoints, obstacles, reward shaping).
- Swap SAC for TD3/PPO by changing the import & model instantiation.
"""
from __future__ import annotations
import math
import time
import argparse
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
except Exception:
    SAC = None  # type: ignore

# ==========================
# Utility math helpers
# ==========================

def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def rot2d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


# ==========================
# Track definition (closed loop)
# ==========================

def catmull_rom_spline(P: np.ndarray, n_samples: int, closed: bool = True, alpha: float = 0.5) -> np.ndarray:
    """Return samples along a Catmull–Rom spline through control points P.
    alpha=0.5 (centripetal) avoids looping/self-intersections.
    If closed, wrap the endpoints to make a loop.
    """
    P = np.asarray(P, dtype=np.float32)
    if closed:
        pts = np.vstack([P[-1:], P, P[:2]])
    else:
        pts = np.vstack([P[0:1], P, P[-1:]])

    def tj(ti, pi, pj):
        return ((np.linalg.norm(pj - pi) ** alpha) + ti)

    samples = []
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        t0 = 0.0
        t1 = tj(t0, p0, p1)
        t2 = tj(t1, p1, p2)
        t3 = tj(t2, p2, p3)
        ts = np.linspace(t1, t2, n_samples, endpoint=False)
        for t in ts:
            A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
            B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
            C  = (t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2
            samples.append(C)
    return np.asarray(samples, dtype=np.float32)


def make_gp_track_waypoints() -> np.ndarray:
    """A full "grand prix"-style closed circuit with hairpins, chicanes, and straights.
    Units are meters. Control points roughly outline a ~1.5 km track.
    """
    ctrl = np.array([
        [  0,   0],   # long start/finish straight
        [200,  10],
        [380,  15],
        [520,  30],   # gentle right kink
        [650, 120],   # braking zone into hairpin
        [580, 220],   # tight hairpin left
        [460, 300],
        [320, 330],
        [200, 320],
        [120, 260],   # medium right
        [ 80, 180],
        [ 60,  90],   # chicane entry
        [120,  40],
        [210,  30],   # exit
        [340,  50],
        [460, 100],
        [560, 180],
        [620, 270],   # fast left sweeper
        [560, 360],
        [440, 440],
        [300, 470],
        [160, 440],
        [ 60, 360],
        [ 10, 250],
        [-10, 130],   # bring back to start straight
    ], dtype=np.float32)
    # Sample dense waypoints
    wp = catmull_rom_spline(ctrl, n_samples=20, closed=True, alpha=0.5)
    # Uniformize spacing (resample at ~5 m spacing)
    diffs = np.diff(np.vstack([wp, wp[0]]), axis=0)
    segs = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(segs[:-1])])
    total = float(np.sum(segs))
    s_query = np.linspace(0.0, total, int(total / 5.0), endpoint=False)
    # linear resample
    x = np.interp(s_query, s, wp[:, 0])
    y = np.interp(s_query, s, wp[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def make_loop_waypoints(n: int = 200, radius_x: float = 30.0, radius_y: float = 20.0, center=(0.0, 0.0)) -> np.ndarray:
    """Simple oval track as fallback."""
    cx, cy = center
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = cx + radius_x * np.cos(t)
    y = cy + radius_y * np.sin(t)
    return np.stack([x, y], axis=1).astype(np.float32).astype(np.float32)


def arclength_param(pts: np.ndarray) -> Tuple[np.ndarray, float]:
    diffs = np.diff(np.vstack([pts, pts[0]]), axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_len[:-1])])
    total = float(np.sum(seg_len))
    return s.astype(np.float32), total


def nearest_on_track(pos: np.ndarray, waypoints: np.ndarray) -> Tuple[int, float]:
    d2 = np.sum((waypoints - pos) ** 2, axis=1)
    idx = int(np.argmin(d2))
    return idx, float(math.sqrt(d2[idx]))


# ==========================
# Environment
# ==========================
@dataclass
class CarParams:
    wheelbase: float = 2.5
    max_steer: float = math.radians(35)
    max_speed: float = 25.0  # m/s ~ 56 mph
    car_radius: float = 1.0


@dataclass
class EnvParams:
    dt: float = 0.05
    lane_width: float = 6.0
    obs_radius: float = 1.0
    n_obstacles: int = 6
    max_steps: int = 1000
    progress_reward_scale: float = 1.0
    lateral_penalty_scale: float = 2.0
    heading_penalty_scale: float = 0.5
    control_penalty_scale: float = 0.01
    speed_penalty_scale: float = 0.01
    collision_penalty: float = 200.0
    offroad_penalty: float = 50.0
    goal_bonus: float = 50.0




class AVCityEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, render_mode: Optional[str] = None, seed: Optional[int] = None, track: str = "gp"):
        super().__init__()
        self.cp = CarParams()
        self.ep = EnvParams()
        self.rng = np.random.RandomState(seed)

        # Track
        if track == "gp":
            self.waypoints = make_gp_track_waypoints()
        else:
            self.waypoints = make_loop_waypoints()

        self.s_track, self.track_len = arclength_param(self.waypoints)

        # Observation: [vx, vy, yaw, steer, lat_err, head_err, s_progress, dist_goal, rel_goal_x, rel_goal_y,
        #               nearest_obs_dist, nearest_obs_angle]
        high = np.array([
            self.cp.max_speed, self.cp.max_speed, np.pi, self.cp.max_steer,
            self.ep.lane_width, np.pi, self.track_len, 1000.0, 1000.0, 1000.0,
            1000.0, np.pi
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Action: [accel, steering_rate] in [-1, 1] scaled internally
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

     
        # Rendering
        self.render_mode = render_mode
        self.pygame_screen = None

        # Auto-fit state (instead of hard-coded scale)
        self._surf_scale = 1.0
        self._origin = np.array([0.0, 0.0], dtype=np.float32)
        self._margin_px = 60.0          # margin around the track
        self._last_surface_size = None  # remember last window size


        # Make sure pygame is initialized early for Windows
        if self.render_mode == "human" and not pygame.get_init():
            pygame.init()

        self.reset(seed=seed)

    # --------------- Gym API ---------------
    def seed(self, seed: Optional[int] = None):
        self.rng.seed(seed)

    def _sample_obstacles(self) -> np.ndarray:
        # Place obstacles along the track with lateral offset inside lane
        idxs = self.rng.choice(len(self.waypoints), size=self.ep.n_obstacles, replace=False)
        obs = []
        for i in idxs:
            center = self.waypoints[i]
            # compute local normal to shift laterally
            prev_pt = self.waypoints[i - 1]
            tangent = center - prev_pt
            tangent /= (np.linalg.norm(tangent) + 1e-8)
            normal = np.array([-tangent[1], tangent[0]])
            lat = self.rng.uniform(-self.ep.lane_width * 0.35, self.ep.lane_width * 0.35)
            pos = center + normal * lat
            obs.append(pos)
        return np.array(obs, dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.seed(seed)
        self.steps = 0
        # start near a waypoint with small noise
        start_idx = int(self.rng.randint(0, len(self.waypoints)))
        start_pos = self.waypoints[start_idx] + self.rng.randn(2).astype(np.float32) * 0.5
        next_idx = (start_idx + 5) % len(self.waypoints)
        start_yaw = math.atan2(self.waypoints[next_idx, 1] - start_pos[1], self.waypoints[next_idx, 0] - start_pos[0])

        self.state = {
            "x": float(start_pos[0]),
            "y": float(start_pos[1]),
            "yaw": float(start_yaw),
            "steer": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "s": float(self.s_track[start_idx]),  # progress along track
        }

        self.goal_idx = int((start_idx + len(self.waypoints) // 4) % len(self.waypoints))  # quarter lap ahead
        self.obstacles = self._sample_obstacles()
        obs = self._get_obs()
        info = {}
        return obs, info

    # --------------- Core dynamics and features ---------------
    def _closest_track_point(self, pos: np.ndarray) -> Tuple[int, float, np.ndarray, np.ndarray]:
        idx, d = nearest_on_track(pos, self.waypoints)
        prev_pt = self.waypoints[idx - 1]
        pt = self.waypoints[idx]
        tangent = pt - prev_pt
        tangent /= (np.linalg.norm(tangent) + 1e-8)
        normal = np.array([-tangent[1], tangent[0]])
        # signed lateral error
        lat_vec = pos - pt
        lat_err = float(np.dot(lat_vec, normal))
        return idx, d, tangent, normal

    def _nearest_obstacle_polar(self, pos: np.ndarray, yaw: float) -> Tuple[float, float]:
        if len(self.obstacles) == 0:
            return 1000.0, 0.0
        rel = self.obstacles - pos
        dists = np.linalg.norm(rel, axis=1)
        k = int(np.argmin(dists))
        rel_body = rot2d(-yaw) @ rel[k]
        angle = math.atan2(rel_body[1], rel_body[0])
        return float(dists[k]), float(angle)

    def _goal_features(self, pos: np.ndarray, yaw: float) -> Tuple[float, float, float]:
        g = self.waypoints[self.goal_idx]
        rel = g - pos
        dist = float(np.linalg.norm(rel))
        rel_body = rot2d(-yaw) @ rel
        return dist, float(rel_body[0]), float(rel_body[1])

    def _get_obs(self) -> np.ndarray:
        x, y, yaw, steer, vx, vy, s = (self.state[k] for k in ["x", "y", "yaw", "steer", "vx", "vy", "s"])
        pos = np.array([x, y], dtype=np.float32)
        idx, _, tangent, normal = self._closest_track_point(pos)
        head_err = wrap_angle(math.atan2(tangent[1], tangent[0]) - yaw)
        lat_err = float(np.dot(pos - self.waypoints[idx], normal))
        dist_goal, rel_goal_x, rel_goal_y = self._goal_features(pos, yaw)
        obs_d, obs_a = self._nearest_obstacle_polar(pos, yaw)

        return np.array([
            vx, vy, yaw, steer,
            lat_err, head_err,
            s, dist_goal, rel_goal_x, rel_goal_y,
            obs_d, obs_a
        ], dtype=np.float32)

    # --------------- Step ---------------
    def step(self, action: np.ndarray):
        self.steps += 1
        a_throttle = float(np.clip(action[0], -1.0, 1.0))  # accel command
        a_steer_rate = float(np.clip(action[1], -1.0, 1.0))

        # Scale controls
        accel = a_throttle * 3.0  # m/s^2
        steer_rate = a_steer_rate * math.radians(45)  # rad/s

        # Unpack state
        x, y, yaw, steer, vx, vy, s = (self.state[k] for k in ["x", "y", "yaw", "steer", "vx", "vy", "s"])
        dt = self.ep.dt

        # Bicycle kinematics with simple dynamics on steer
        steer = float(np.clip(steer + steer_rate * dt, -self.cp.max_steer, self.cp.max_steer))
        speed = float(np.clip(math.hypot(vx, vy) + accel * dt, 0.0, self.cp.max_speed))
        beta = math.atan(0.5 * math.tan(steer))  # simple slip approximation
        yaw = wrap_angle(yaw + (speed / self.cp.wheelbase) * math.sin(2 * beta) * dt)
        vx = speed * math.cos(yaw)
        vy = speed * math.sin(yaw)
        x += vx * dt
        y += vy * dt

        # Update progress s by projecting movement onto local tangent
        pos = np.array([x, y], dtype=np.float32)
        idx, _, tangent, _ = self._closest_track_point(pos)
        ds = float(np.dot(np.array([vx, vy], dtype=np.float32) * dt, tangent))
        s = (s + max(ds, 0.0)) % self.track_len

        # Write back state
        self.state.update({"x": x, "y": y, "yaw": yaw, "steer": steer, "vx": vx, "vy": vy, "s": s})

        # Compute costs & rewards
        obs = self._get_obs()
        lat_err = abs(obs[4])
        head_err = abs(obs[5])
        progress = max(ds, 0.0)
        speed_pen = speed
        ctrl_pen = a_throttle ** 2 + a_steer_rate ** 2

        reward = (
            self.ep.progress_reward_scale * progress
            - self.ep.lateral_penalty_scale * lat_err
            - self.ep.heading_penalty_scale * head_err
            - self.ep.control_penalty_scale * ctrl_pen
            - self.ep.speed_penalty_scale * speed_pen * (lat_err > self.ep.lane_width * 0.25)
        )

        terminated = False
        truncated = False

        # Off-road (beyond lane boundaries)
        if lat_err > self.ep.lane_width * 0.5:
            reward -= self.ep.offroad_penalty
            terminated = True

        # Collision check
        dists = np.linalg.norm(self.obstacles - pos, axis=1) if len(self.obstacles) else np.array([np.inf])
        if np.any(dists < (self.cp.car_radius + self.ep.obs_radius)):
            reward -= self.ep.collision_penalty
            terminated = True

        # Reached goal window
        goal_pos = self.waypoints[self.goal_idx]
        if np.linalg.norm(goal_pos - pos) < 3.0:
            reward += self.ep.goal_bonus
            # advance goal further along the track
            self.goal_idx = (self.goal_idx + len(self.waypoints) // 4) % len(self.waypoints)

        if self.steps >= self.ep.max_steps:
            truncated = True

        info = {"progress": progress}
        return obs, float(reward), terminated, truncated, info


    def _fit_track_to_surface(self, surface_size):
        """Compute scale and origin so the whole track fits in the window with a margin."""
        w, h = surface_size
        mins = self.waypoints.min(axis=0)
        maxs = self.waypoints.max(axis=0)
        size = np.maximum(maxs - mins, 1e-6)  # avoid zero

        usable_w = max(1.0, w - 2 * self._margin_px)
        usable_h = max(1.0, h - 2 * self._margin_px)

        sx = usable_w / float(size[0])
        sy = usable_h / float(size[1])
        self._surf_scale = float(min(sx, sy))

        # center the track
        world_center_px = (mins + 0.5 * size) * self._surf_scale
        screen_center = np.array([w * 0.5, h * 0.5], dtype=np.float32)
        self._origin = screen_center - world_center_px

        self._last_surface_size = (w, h)


    # --------------- Rendering ---------------
    def render(self):
        if self.render_mode is None:
            return

        if self.pygame_screen is None:
            pygame.display.quit()
            pygame.display.init()
            w, h = 1280, 720
            # Make the window resizable
            self.pygame_screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
            pygame.display.set_caption("AVCityEnv")
            # Fit the track to the initial window
            self._fit_track_to_surface((w, h))

        screen = self.pygame_screen

        # Handle events (including resize)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            elif event.type == pygame.VIDEORESIZE:
                # Refit to the new window size
                self._fit_track_to_surface(screen.get_size())

        # If size changed (e.g., DPI/OS moves), refit
        curr_size = screen.get_size()
        if self._last_surface_size != curr_size:
            self._fit_track_to_surface(curr_size)

        screen.fill((240, 240, 240))

        # world -> screen
        def W(p):
            q = (p * self._surf_scale) + self._origin
            return (int(q[0]), int(q[1]))

        # draw track centerline
        pts = ((self.waypoints * self._surf_scale) + self._origin).astype(np.int32)
        if len(pts) > 1:
            pygame.draw.lines(screen, (80, 80, 80), True, pts.tolist(), 2)

        # draw lane boundaries
        for side in (-1, 1):
            edge_pts = []
            for i in range(0, len(self.waypoints), 3):
                prev_pt = self.waypoints[i - 1]
                pt = self.waypoints[i]
                t = pt - prev_pt
                t /= (np.linalg.norm(t) + 1e-8)
                n = np.array([-t[1], t[0]])
                edge = pt + n * (side * self.ep.lane_width * 0.5)
                edge_pts.append(W(edge))
            if len(edge_pts) > 1:
                pygame.draw.lines(screen, (180, 180, 180), True, edge_pts, 2)

        # draw obstacles
        for o in self.obstacles:
            pygame.draw.circle(screen, (200, 60, 60), W(o), int(self.ep.obs_radius * self._surf_scale))

        # draw goal
        pygame.draw.circle(screen, (60, 140, 200), W(self.waypoints[self.goal_idx]), 8)

        # draw car
        x, y, yaw, steer = (self.state[k] for k in ["x", "y", "yaw", "steer"])
        car_pos = np.array([x, y], dtype=np.float32)
        car_px = W(car_pos)
        pygame.draw.circle(screen, (60, 60, 220), car_px, int(self.cp.car_radius * self._surf_scale))
        tip = car_pos + rot2d(yaw) @ np.array([2.0, 0.0], dtype=np.float32)
        pygame.draw.line(screen, (20, 20, 120), car_px, W(tip), 3)

        pygame.display.flip()
        time.sleep(1.0 / self.metadata["render_fps"])


    def close(self):
        if self.pygame_screen is not None:
            pygame.display.quit()
            self.pygame_screen = None
            pygame.quit()


# ==========================
# Training / Evaluation
# ==========================
from stable_baselines3.common.callbacks import BaseCallback
import os
import pandas as pd
import matplotlib.pyplot as plt

class DemoRenderCallback(BaseCallback):
    """Periodically run a short deterministic rollout with rendering in a separate env."""
    def __init__(self, render_every_steps: int = 0, demo_steps: int = 300, track: str = "gp"):
        super().__init__()
        self.render_every_steps = render_every_steps
        self.demo_steps = demo_steps
        self._next = render_every_steps
        self._track = track

    def _on_step(self) -> bool:
        if self.render_every_steps and self.num_timesteps >= self._next:
            env = AVCityEnv(render_mode="human", track=self._track)
            obs, _ = env.reset()
            done = False
            trunc = False
            steps = 0
            while steps < self.demo_steps and not (done or trunc):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, trunc, _ = env.step(action)
                env.render()
                steps += 1
            env.close()
            self._next += self.render_every_steps
        return True


def plot_rewards(log_dir: str = "./logs", out_path: str = "reward_curve.png", window: int = 50):
    files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith(".monitor.csv") or f == "monitor.csv"]
    if not files:
        print(f"No monitor.csv found in {log_dir}")
        return
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, comment="#")
            dfs.append(df)
        except Exception:
            pass
    if not dfs:
        print("No readable monitor files.")
        return
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("l").reset_index(drop=True) if "l" in df.columns else df
    df["rolling_return"] = df["r"].rolling(window=window, min_periods=1).mean()

    plt.figure()
    plt.plot(df.index + 1, df["r"], alpha=0.3, label="episode return")
    plt.plot(df.index + 1, df["rolling_return"], label=f"rolling mean ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Training rewards")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved reward plot to {out_path}")


def make_env(render_mode: Optional[str] = None, track: str = "gp"):
    def _fn():
        env = AVCityEnv(render_mode=render_mode, track=track)
        return env
    return _fn


def train(total_timesteps: int = 200_000, log_dir: str = "./logs", model_path: str = "./sac_avcity.zip", render_every_steps: int = 0, track: str = "gp"):
    if SAC is None:
        raise RuntimeError("Stable-Baselines3 is required: pip install stable-baselines3[extra]")

    os.makedirs(log_dir, exist_ok=True)
    env = DummyVecEnv([lambda: Monitor(AVCityEnv(render_mode=None, track=track), filename=os.path.join(log_dir, "monitor.csv"))])
    eval_env = DummyVecEnv([lambda: Monitor(AVCityEnv(render_mode=None, track=track), filename=os.path.join(log_dir, "eval_monitor.csv"))])

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        gamma=0.99,
        tau=0.02,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    ckpt_cb = CheckpointCallback(save_freq=50_000, save_path=log_dir, name_prefix="sac_avcity_ckpt")
    demo_cb = DemoRenderCallback(render_every_steps=render_every_steps, demo_steps=300, track=track)

    model.learn(total_timesteps=total_timesteps, callback=[eval_cb, ckpt_cb, demo_cb])
    model.save(model_path)
    env.close(); eval_env.close()


def evaluate(model_path: str = "./sac_avcity.zip", episodes: int = 5, track: str = "gp"):
    if SAC is None:
        raise RuntimeError("Stable-Baselines3 is required: pip install stable-baselines3[extra]")

    env = AVCityEnv(render_mode="human", track=track)
    model = SAC.load(model_path)
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        trunc = False
        ep_ret = 0.0
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)
            ep_ret += reward
            env.render()
        print(f"Episode {ep+1}: return={ep_ret:.2f}")
    env.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--model", type=str, default="./sac_avcity.zip")
    parser.add_argument("--render_train_every", type=int, default=0,
                        help="Render a short demo during training every N steps (0=off)")
    parser.add_argument("--plot", action="store_true",
                        help="Plot reward curves from logs/monitor.csv")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--track", type=str, default="gp", choices=["gp", "oval"],
                        help="Choose track layout")
    args = parser.parse_args()

    if args.train:
        train(total_timesteps=args.timesteps,
              model_path=args.model,
              log_dir=args.log_dir,
              render_every_steps=args.render_train_every,
              track=args.track)
    elif args.eval:
        evaluate(model_path=args.model, track=args.track)
    elif args.plot:
        plot_rewards(log_dir=args.log_dir)
    else:
        print(__doc__)


 



## python self_driving_rl_gym_env.py --train --track gp --timesteps 1000 --model ./models/sac_avcity.zip
## python self_driving_rl_gym_env.py --eval --track gp --model ./models/gp_run1/sac_avcity.zip
