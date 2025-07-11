# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Joystick task for the BD-5
# See Booster T1 in the locomotion folder for more details
# And OpenDuck for value range

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import gait
from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding
from mujoco_playground._src.locomotion.bd5 import base as bd5_base
from mujoco_playground._src.locomotion.bd5 import bd5_constants as consts

CLIP_MOTOR_SPEED = True

def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.002,
      episode_length=1000,
      action_repeat=1,
      action_scale=0.3, # OK
      dof_vel_scale=1.0,
      history_len=1,
      soft_joint_pos_limit_factor=0.95,
      max_motor_velocity=4.50, # clip motor velocity at 4.82 at 12.0V or 3.8 nominal at 11.1V
      noise_config=config_dict.create(
          level=1.0,  # Set to 0.0 to disable noise.
          action_min_delay=0,  # env steps
          action_max_delay=3,  # env steps 
          obs_min_delay=0,  # env steps
          obs_max_delay=1,  # env steps
          scales=config_dict.create(
              hip_pos=0.03,
              knee_pos=0.03,
              ankle_pos=0.03,
              joint_vel=1.5, # 1.5
              gravity=0.05, # 0.1
              linvel=0.1,
              gyro=0.2,
              accelerometer=0.2,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Tracking related reward
              tracking_lin_vel=4.0, # follow the joystick command x, y 1.0
              tracking_ang_vel=1.0, # follow the joystick command theta 0.5 or 0.8
              # Base related rewards.
              lin_vel_z=0.0, # OK -2.0 or 0.0
              ang_vel_xy=-0.15, # OK -0.15 or -0.05
              orientation=-5.0, # OK -5.0
              base_height=0.0, # OK 0.0
              # Energy related rewards.
              torques=0.0, # penalize high torques -0.0002
              action_rate=0.0, # penalize rapid changes in action -0.001
              energy=0.0, # penalize ernergy consumption -0.0001 / -2e-5
              # Feet related rewards.
              feet_clearance=0.0, # -> was -0.5
              feet_air_time=2.0, # was 2.0 
              feet_slip=-0.25, # was -0.25 similar to feet_drag 
              feet_drag=0.0, # was -1.5 
              feet_height=0.0,
              feet_phase=1.0, # was 1.0 
              foot_lift_reward=0.0, # reward when z_vel > 0 during swing 
              swing_symmetry=0.0, # encourage symmetric gait 
              # Other rewards.
              stand_still=0.0, # penalize when command = 0
              alive=2.0,
              termination=-1.0,
              # Pose related rewards.
              joint_deviation_ankle=0.0, # was -0.25
              joint_deviation_knee=-0.1, # was -0.1
              joint_deviation_hip=-0.1, # was -0.25
              dof_pos_limits=-1.0,
              pose=-1.0, # TEST IT (was -1.0) 
          ),
          tracking_sigma=0.25, # OK
          max_foot_height=0.05,
          base_height_target=0.224,
      ),
      push_config=config_dict.create(
          enable=True,
          interval_range=[5.0, 10.0],
          magnitude_range=[0.1, 2.0],
      ),
        lin_vel_x=[-0.15, 0.15],
        lin_vel_y=[-0.2, 0.2],
        ang_vel_yaw=[-0.8, 0.8],
  )

class Joystick(bd5_base.BD5Env):
    """Track a joystick command."""
    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=consts.task_to_xml(task).as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        # Initialize
        self._post_init()

    def _post_init(self) -> None:
        # Init default pose
        self._init_q = jp.array(self._mj_model.keyframe("init_pose").qpos)
        print(self._init_q)
        self._default_pose = jp.array(self._mj_model.keyframe("init_pose").qpos[7:])

        # Get the range of the joints
        # Note: First joint is freejoint.
        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

        hip_ids = [idx for idx, j in enumerate(consts.JOINTS_ORDER) if "_hip" in j]
        knee_ids = [idx for idx, j in enumerate(consts.JOINTS_ORDER) if "_knee" in j]
        ankle_ids = [idx for idx, j in enumerate(consts.JOINTS_ORDER) if "_ankle" in j]

        self._hip_indices = jp.array(hip_ids)
        self._knee_indices = jp.array(knee_ids)
        self._ankle_indices = jp.array(ankle_ids)

        # fmt: off
        self._weights = jp.array(
            [
                2.0, # left_hip_yaw 0
                1.0, # left_hip_roll 1
                0.01, # left_hip_pitch 2
                0.01, # left_knee 3
                1.0, # left_ankle 4
                2.0, # right_hip_yaw 5 
                1.0, # right_hip_roll 6
                0.01, # right_hip_pitch 7
                0.01, # right_knee 8
                1.0, # right_ankle 9
            ]
        )
        # fmt: on

        self._nb_joints = self._mj_model.njnt # number of joints
        self._nb_actuators = self._mj_model.nu # number of actuators
        print("Number of Joints and Actuators =", self._nb_joints, self._nb_actuators)

        self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
        self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
        self._site_id = self._mj_model.site("imu").id
        print("BD-5 mass =", self._torso_mass)

        self._feet_site_id = np.array([self._mj_model.site(name).id for name in consts.FEET_SITES])
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._feet_geom_id = np.array([self._mj_model.geom(name).id for name in consts.FEET_GEOMS])

        foot_linvel_sensor_adr = []
        for site in consts.FEET_LINVEL:
            sensor_id = self._mj_model.sensor(f"{site}").id
            sensor_adr = self._mj_model.sensor_adr[sensor_id]
            sensor_dim = self._mj_model.sensor_dim[sensor_id]
            foot_linvel_sensor_adr.append(list(range(sensor_adr, sensor_adr + sensor_dim)))
        self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

        # Joint noise scale
        qpos_noise_scale = np.zeros(self._nb_actuators)
        qpos_noise_scale[hip_ids] = self._config.noise_config.scales.hip_pos
        qpos_noise_scale[knee_ids] = self._config.noise_config.scales.knee_pos
        qpos_noise_scale[ankle_ids] = self._config.noise_config.scales.ankle_pos
        self._qpos_noise_scale = jp.array(qpos_noise_scale)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        # Init position / velocity state 
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)

        # x=+U(-0.05, 0.05), y=+U(-0.05, 0.05), yaw=U(-3.14, 3.14).
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.05, maxval=0.05)

        # floating base [x, y ,z, a, b, c, d] = 7
        qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        new_quat = math.quat_mul(qpos[3:7], quat)
        qpos = qpos.at[3:7].set(new_quat)

        # Joints qpos[7:]=*U(0.5, 1.5)
        rng, key = jax.random.split(rng)
        qpos = qpos.at[7:].set(qpos[7:] * jax.random.uniform(key, (self._nb_actuators,), minval=0.5, maxval=1.5))

        # Floating base init velocity d(xyzrpy)=U(-0.5, 0.5)
        rng, key = jax.random.split(rng)
        qvel = qvel.at[0:6].set(jax.random.uniform(key, (6,), minval=-0.05, maxval=0.05))

        # Initialize
        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:])

        # Phase, freq=U(1.25, 1.5)
        rng, key = jax.random.split(rng)
        gait_freq = jax.random.uniform(key, (1,), minval=1.25, maxval=1.75)
        phase_dt = 2 * jp.pi * self.dt * gait_freq
        phase = jp.array([0, jp.pi])

        # Init input command
        rng, cmd_rng = jax.random.split(rng)
        cmd = self.sample_command(cmd_rng)

        # Sample push interval.
        rng, push_rng = jax.random.split(rng)
        push_interval = jax.random.uniform(
            push_rng,
            minval=self._config.push_config.interval_range[0],
            maxval=self._config.push_config.interval_range[1],
        )
        push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

        info = {
            "rng": rng,
            "step": 0,
            "command": cmd,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "last_last_last_act": jp.zeros(self.mjx_model.nu),
            "motor_targets": self._default_pose,
            "feet_air_time": jp.zeros(2),
            "last_contact": jp.zeros(2, dtype=bool),
            "swing_peak": jp.zeros(2),
            # Phase related.
            "phase_dt": phase_dt,
            "phase": phase,
            # Push related.
            "push": jp.array([0.0, 0.0]),
            "push_step": 0,
            "push_interval_steps": push_interval_steps,
            # History related.
            "action_history": jp.zeros(self._config.noise_config.action_max_delay * self._nb_actuators),
            "obs_history": jp.zeros(self._config.noise_config.obs_max_delay * 66) # Eww hardcoded !
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())
        metrics["swing_peak"] = jp.zeros(())

        contact = jp.array([geoms_colliding(data, geom_id, self._floor_geom_id) for geom_id in self._feet_geom_id])

        obs = self._get_obs(data, info, contact)
        reward, done = jp.zeros(2)

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # Split the rng seed
        state.info["rng"], push1_rng, push2_rng, action_delay_rng = jax.random.split(state.info["rng"], 4)

        # Handle action delay
        action_history = (
            jp.roll(state.info["action_history"], self._nb_actuators)
            .at[: self._nb_actuators]
            .set(action)
        )
        state.info["action_history"] = action_history
        action_idx = jax.random.randint(
            action_delay_rng,
            (1,),
            minval=self._config.noise_config.action_min_delay,
            maxval=self._config.noise_config.action_max_delay,
        )
        action_w_delay = action_history.reshape((-1, self._nb_actuators))[
            action_idx[0]
        ]  # action with delay

        # Push at the free joint -> eg. body
        push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
        push_magnitude = jax.random.uniform(
            push2_rng,
            minval=self._config.push_config.magnitude_range[0],
            maxval=self._config.push_config.magnitude_range[1],
        )
        push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
        push *= jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"]) == 0
        push *= self._config.push_config.enable

        qvel = state.data.qvel
        qvel = qvel.at[:2].set(push * push_magnitude + qvel[:2])
        data = state.data.replace(qvel=qvel)
        state = state.replace(data=data)

        # Motor targets with action delay and noise
        motor_targets = self._default_pose + action_w_delay * self._config.action_scale

        # Clip motor speed if needed
        if CLIP_MOTOR_SPEED:
            prev_motor_targets = state.info["motor_targets"]
            motor_targets = jp.clip(motor_targets, 
                                    prev_motor_targets - self._config.max_motor_velocity * self.dt, 
                                    prev_motor_targets + self._config.max_motor_velocity * self.dt
                                    )

        # env step 
        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)
        # memory update 
        state.info["motor_targets"] = motor_targets

        # Contact phase
        contact = jp.array([geoms_colliding(data, geom_id, self._floor_geom_id) for geom_id in self._feet_geom_id])
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        state.info["feet_air_time"] += self.dt
        p_f = data.site_xpos[self._feet_site_id]
        p_fz = p_f[..., -1]
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)
    
        # Update state
        obs = self._get_obs(data, state.info, contact)
        done = self._get_termination(data)

        # Reward
        rewards = self._get_reward(data, action, state.info, state.metrics, done, first_contact, contact)
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        state.info["push"] = push
        state.info["step"] += 1
        state.info["push_step"] += 1
        phase_tp1 = state.info["phase"] + state.info["phase_dt"]
        state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi
        state.info["phase"] = jp.where(
            jp.linalg.norm(state.info["command"]) > 0.01,
            state.info["phase"],
            jp.ones(2) * jp.pi,
        )
        state.info["last_last_last_act"] = state.info["last_last_act"]
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
        state.info["command"] = jp.where(
            state.info["step"] > 500,
            self.sample_command(cmd_rng),
            state.info["command"],
        )
        state.info["step"] = jp.where(
            done | (state.info["step"] > 500),
            0,
            state.info["step"],
        )
        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= ~contact
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state
    
    def _get_termination(self, data: mjx.Data) -> jax.Array:
        fall_termination = self.get_gravity(data)[-1] < 0.0
        return fall_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    
    def _get_single_obs(self, data: mjx.Data, info: dict[str, Any], contact: jax.Array) -> mjx_env.Observation:
        # Noisy gyro -> raw gyro measurement IMU sensor
        gyro = self.get_gyro(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gyro = (
            gyro
            + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gyro
        )

        # Noisy accelerometer -> raw accelerometer measurement IMU sensor
        accelerometer = self.get_accelerometer(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_accelerometer = (
            accelerometer
            + (2 * jax.random.uniform(noise_rng, shape=accelerometer.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.accelerometer
        )

        # Noisy gravity vector -> use fused IMU measurement 
        gravity = data.site_xmat[self._site_id].T @ jp.array([0, 0, -1])
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gravity = (
            gravity
            + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gravity
        )

        # Noisy joint angle -> dynamixel position 
        joint_angles = data.qpos[7:]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = (
            joint_angles
            + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
            * self._config.noise_config.level
            * self._qpos_noise_scale
        )

        # Noisy joint velocity -> dynamixel velocity
        joint_vel = data.qvel[6:]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_vel = (
            joint_vel
            + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_vel
        )

        # Step phase -> identical on the real robot 
        cos = jp.cos(info["phase"])
        sin = jp.sin(info["phase"])
        phase = jp.concatenate([cos, sin])

        # Real robot state observation n=63
        state = jp.hstack(
            [
                noisy_gyro,  # 3 (gx, gy, gz)
                noisy_accelerometer,  # 3 (ax, ay, az)
                noisy_gravity, # 3 (grx, gry, grz)
                info["command"],  # 3 (Vx, Vy, Vyaw)
                noisy_joint_angles - self._default_pose,  # NUM_JOINTS
                noisy_joint_vel * self._config.dof_vel_scale,  # NUM_JOINTS
                info["last_act"],  # NUM_JOINTS
                info["last_last_act"],  # NUM_JOINTS
                info["last_last_last_act"],  # NUM_JOINTS
                phase, # 4 (cos(th1, th2), sin(th1, th2)) # TODO : add back the phase if needed
            ]
        )

        # Prileged obsevation -> non-noisy state observation
        linvel = self.get_local_linvel(data)
        global_angvel = self.get_global_angvel(data)
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
        root_height = data.qpos[2]

        # Simulated robot full state observation
        privileged_state = jp.hstack(
            [
                state,
                gyro,  # 3
                accelerometer,  # 3
                gravity,  # 3
                linvel,  # 3
                global_angvel,  # 3
                joint_angles - self._default_pose, # NUM_JOINTS
                joint_vel, # NUM_JOINTS
                root_height,  # 1
                data.actuator_force,  # NUM_JOINTS
                contact,  # 2
                feet_vel,  # 4*3
                info["feet_air_time"],  # 2
            ]
        )

        return {
            "state": state,
            "privileged_state": privileged_state,
        }
    
    def _get_obs(self, data: mjx.Data, info: dict[str, Any], contact: jax.Array) -> mjx_env.Observation:
        obs = self._get_single_obs(data, info, contact)
        # edit only the true robot state 
        obs_size = obs["state"].shape[0]

        # fill the buffer
        obs_history = (
            jp.roll(info["obs_history"], obs_size).at[:obs_size].set(obs["state"])
        )
        info["obs_history"] = obs_history

        # add observation delay
        info["rng"], key = jax.random.split(info["rng"])
        obs_delay_idx = jax.random.randint(
            key,
            (1,),
            minval=self._config.noise_config.obs_min_delay,
            maxval=self._config.noise_config.obs_max_delay,
        )
        obs["state"] = obs_history.reshape((-1, obs_size))[obs_delay_idx[0]]
        # delayed observation
        return obs

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
        first_contact: jax.Array,
        contact: jax.Array,
    ) -> dict[str, jax.Array]:
        del metrics  # Unused.
        return {
            # Tracking rewards.
            "tracking_lin_vel": self._reward_tracking_lin_vel(info["command"], self.get_local_linvel(data)),
            "tracking_ang_vel": self._reward_tracking_ang_vel(info["command"], self.get_gyro(data)),
            # Base-related rewards.
            "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
            "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
            "orientation": self._cost_orientation(self.get_gravity(data)),
            "base_height": self._cost_base_height(data.qpos[2]),
            # Energy related rewards.
            "torques": self._cost_torques(data.actuator_force),
            "action_rate": self._cost_action_rate(action, info["last_act"], info["last_last_act"]),
            "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
            # Feet related rewards.
            "feet_clearance": self._cost_feet_clearance(data, info),
            "feet_slip": self._cost_feet_slip(data, contact, info),
            "feet_drag": self._reward_feet_drag(data, contact),
            "feet_height": self._cost_feet_height(info["swing_peak"], first_contact, info),
            "feet_air_time": self._reward_feet_air_time(info["feet_air_time"], first_contact, info["command"]),
            "feet_phase": self._reward_feet_phase(
                data,
                info["phase"],
                self._config.reward_config.max_foot_height,
                info["command"],
            ),
            "swing_symmetry": self._reward_swing_symmetry(data, contact, info["command"]),
            "foot_lift_reward": self._reward_foot_lift_velocity(data, contact, info["command"]),
            # Other rewards.
            "alive": self._reward_alive(),
            "termination": self._cost_termination(done),
            "stand_still": self._cost_stand_still(info["command"], data.qpos[7:], data.qvel[6:]),
            # Pose related rewards.
            "joint_deviation_hip": self._cost_joint_deviation_hip(data.qpos[7:], info["command"]),
            "joint_deviation_knee": self._cost_joint_deviation_knee(data.qpos[7:]),
            "joint_deviation_ankle": self._cost_joint_deviation_ankle(data.qpos[7:]),
            "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
            "pose": self._cost_pose(data.qpos[7:]),
        }

    def _reward_tracking_lin_vel(
        self,
        commands: jax.Array,
        local_vel: jax.Array,
    ) -> jax.Array:
        y_tol = 0.1
        error_x = jp.square(commands[0] - local_vel[0])
        error_y = jp.clip(jp.abs(local_vel[1] - commands[1]) - y_tol, 0.0, None)
        lin_vel_error = error_x + jp.square(error_y)
        reward = jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)
        return jp.nan_to_num(reward)

    def _reward_tracking_ang_vel(
        self,
        commands: jax.Array,
        ang_vel: jax.Array,
    ) -> jax.Array:
        ang_vel_error = jp.square(commands[2] - ang_vel[2])
        reward = jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)
        return jp.nan_to_num(reward)

    # Base-related rewards.
    def _cost_lin_vel_z(self, global_linvel: jax.Array) -> jax.Array:
        return jp.nan_to_num(jp.square(global_linvel[2]))

    def _cost_ang_vel_xy(self, global_angvel: jax.Array) -> jax.Array:
        return jp.nan_to_num(jp.sum(jp.square(global_angvel[:2])))

    def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
        return jp.nan_to_num(jp.sum(jp.square(torso_zaxis[:2])))

    def _cost_base_height(self, base_height: jax.Array) -> jax.Array:
        return jp.nan_to_num(jp.square(base_height - self._config.reward_config.base_height_target))

    # Energy related rewards.
    def _cost_torques(self, torques: jax.Array) -> jax.Array:
        return jp.nan_to_num(jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques)))

    def _cost_energy(self, qvel: jax.Array, qfrc_actuator: jax.Array) -> jax.Array:
        return jp.nan_to_num(jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator)))

    def _cost_action_rate(self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array) -> jax.Array:
        # Penalize first derivative of actions.
        c1 = jp.sum(jp.square(act - last_act))
        c2 = jp.sum(jp.square(act - 2 * last_act + last_last_act))
        return jp.nan_to_num(c1 + c2)

    # Other rewards.
    def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
        out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
        out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
        return jp.nan_to_num(jp.sum(out_of_limits))

    def _cost_stand_still(
        self, commands: jax.Array, qpos: jax.Array, qvel: jax.Array 
    ) -> jax.Array:
        cmd_norm = jp.linalg.norm(commands)
        cost_pos = jp.sum(jp.square(qpos - self._default_pose))
        cost_vel = jp.sum(jp.abs(qvel))
        cost = (cost_pos + cost_vel) * (cmd_norm < 0.01)
        return jp.nan_to_num(cost)

    def _cost_termination(self, done: jax.Array) -> jax.Array:
        return done

    def _reward_alive(self) -> jax.Array:
        return jp.array(1.0)

    # Pose-related rewards.
    def _cost_joint_deviation_hip(self, qpos: jax.Array, cmd: jax.Array) -> jax.Array:
        cost = jp.sum(jp.abs(qpos[self._hip_indices] - self._default_pose[self._hip_indices]))
        cost *= jp.abs(cmd[1]) > 0.01
        return jp.nan_to_num(cost)

    def _cost_joint_deviation_knee(self, qpos: jax.Array) -> jax.Array:
        return jp.nan_to_num(jp.sum(jp.abs(qpos[self._knee_indices] - self._default_pose[self._knee_indices])))

    def _cost_joint_deviation_ankle(self, qpos: jax.Array) -> jax.Array:
        return jp.nan_to_num(jp.sum(jp.abs(qpos[self._ankle_indices] - self._default_pose[self._ankle_indices])))

    def _cost_pose(self, qpos: jax.Array) -> jax.Array:
        return jp.nan_to_num(jp.sum(jp.square(qpos - self._default_pose) * self._weights))

    # Feet related rewards.
    """
    def _cost_feet_slip(self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]) -> jax.Array:
        del info  # Unused.
        body_vel = self.get_global_linvel(data)[:2]
        reward = jp.sum(jp.linalg.norm(body_vel, axis=-1) * contact)
        return jp.nan_to_num(reward)
    """
    def _cost_feet_slip(self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]) -> jax.Array:
        del info  # Unused.
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
        vel_xy = feet_vel[..., :2]
        vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
        reward = jp.sum(jp.linalg.norm(vel_xy_norm_sq, axis=-1) * contact)
        return jp.nan_to_num(reward)
    
    def _reward_feet_drag(self, data: mjx.Data, contact: jax.Array) -> jax.Array:
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
        feet_vel_xy = feet_vel[..., :2]
        feet_vel_norm = jp.linalg.norm(feet_vel_xy, axis=-1)
        feet_vel_norm *= contact
        # Normalize with a softmax-like transformation
        return jp.nan_to_num(jp.sum(1.0 - jp.exp(-feet_vel_norm * 10.0)))

    def _cost_feet_clearance(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        del info  # Unused.
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
        vel_xy = feet_vel[..., :2]
        vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
        foot_pos = data.site_xpos[self._feet_site_id]
        foot_z = foot_pos[..., -1]
        delta = (foot_z - self._config.reward_config.max_foot_height) ** 2
        return jp.nan_to_num(jp.sum(delta * vel_norm))

    def _cost_feet_height(
        self,
        swing_peak: jax.Array,
        first_contact: jax.Array,
        info: dict[str, Any],
    ) -> jax.Array:
        del info  # Unused.
        error = swing_peak / self._config.reward_config.max_foot_height - 1.0
        return jp.nan_to_num(jp.sum(jp.square(error) * first_contact))

    def _reward_feet_air_time(
        self,
        air_time: jax.Array,
        first_contact: jax.Array,
        commands: jax.Array,
        threshold_min: float = 0.2,
        threshold_max: float = 0.5,
    ) -> jax.Array:
        del commands
        air_time = (air_time - threshold_min) * first_contact
        air_time = jp.clip(air_time, max=threshold_max - threshold_min)
        reward = jp.sum(air_time)
        reward += -jp.square(air_time[0] - air_time[1]) # TODO : try this ? 
        return jp.nan_to_num(reward)

    def _reward_foot_lift_velocity(
            self, 
            data: mjx.Data,
            contact: jax.Array, 
            commands: jax.Array
        ) -> jax.Array:
        # Encourage upward velocity when foot is not in contact
        cmd_norm = jp.linalg.norm(commands)
        foot_vels = data.sensordata[self._foot_linvel_sensor_adr].reshape(2, 3)
        upward_vel = jp.clip(foot_vels[:, 2], 0.0, 1.0)  # only reward upward Z
        not_in_contact = 1.0 - contact
        mask = jp.clip(cmd_norm / 0.01, 0.0, 1.0)
        return jp.nan_to_num(jp.sum(upward_vel * not_in_contact) * mask)

    def _reward_swing_symmetry(
            self, 
            data: mjx.Data, 
            contact: jax.Array,
            commands: jax.Array
        ) -> jax.Array:
        # Encourage mirroring swing (velocities) when both feet are off ground
        cmd_norm = jp.linalg.norm(commands)
        foot_vels = data.sensordata[self._foot_linvel_sensor_adr].reshape(2, 3)
        swing_phase = (1.0 - contact) > 0.5
        diff = foot_vels[0] + foot_vels[1]  # they should be opposites
        mask = jp.clip(cmd_norm / 0.01, 0.0, 1.0)
        return jp.nan_to_num(jp.where(jp.logical_and(swing_phase[0], swing_phase[1]), -jp.sum(jp.square(diff)), 0.0) * mask)

    def _reward_feet_phase(
        self,
        data: mjx.Data,
        phase: jax.Array,
        foot_height: jax.Array,
        commands: jax.Array,
    ) -> jax.Array:
        # Reward for tracking the desired foot height under phase 
        foot_pos = data.site_xpos[self._feet_site_id]
        foot_z = foot_pos[..., -1]
        rz = gait.get_rz(phase, swing_height=foot_height)
        error = jp.sum(jp.square(foot_z - rz))
        reward = jp.exp(-error / 0.01)
        body_linvel = self.get_global_linvel(data)[:2]
        body_angvel = self.get_global_angvel(data)[2]
        linvel_mask = jp.logical_or(
            jp.linalg.norm(body_linvel) > 0.1,
            jp.abs(body_angvel) > 0.1,
        )
        mask = jp.logical_and(linvel_mask, jp.linalg.norm(commands) > 0.01)
        reward *= mask
        return jp.nan_to_num(reward)

    def sample_command(self, rng: jax.Array) -> jax.Array:
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
        lin_vel_x = jax.random.uniform(rng1, minval=self._config.lin_vel_x[0], maxval=self._config.lin_vel_x[1])
        lin_vel_y = jax.random.uniform(rng2, minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1])
        ang_vel_yaw = jax.random.uniform(rng3, minval=self._config.ang_vel_yaw[0], maxval=self._config.ang_vel_yaw[1])
        # With 10% chance, set everything to zero.
        return jp.where(
            jax.random.bernoulli(rng4, p=0.1),
            jp.zeros(3),
            jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw]),
        )