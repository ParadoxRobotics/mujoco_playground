# Copyright 2024 DeepMind Technologies Limited
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
"""Deploy an MJX policy in ONNX format to C MuJoCo and play with it."""

from etils import epath
import mujoco
import mujoco.viewer as viewer
import numpy as np
import onnxruntime as rt

from mujoco_playground._src.locomotion.bd5 import bd5_constants
from mujoco_playground._src.locomotion.bd5.base import get_assets
from mujoco_playground.experimental.sim2sim.gamepad_reader_bd5 import Gamepad

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE / "onnx"

class OnnxController:
  """ONNX controller for the BD-5."""

  def __init__(
      self,
      policy_path: str,
      default_angles: np.ndarray,
      ctrl_dt: float,
      n_substeps: int,
      action_scale: float = 0.3,
      vel_range_x: float = [-1.0, 1.0],
      vel_range_y: float = [-1.0, 1.0],
      vel_range_rot: float = [-1.0, 1.0],
      gait_freq:float = 1.25,
      max_motor_speed: float = 4.82,
  ):
    self._output_names = ["continuous_actions"]
    self._policy = rt.InferenceSession(
        policy_path, providers=["CPUExecutionProvider"]
    )

    # Init action scale and memory
    self._action_scale = action_scale
    self._default_angles = default_angles
    self._last_action = np.zeros_like(default_angles, dtype=np.float32)
    self._last_last_action = np.zeros_like(default_angles, dtype=np.float32)
    self._last_last_last_action = np.zeros_like(default_angles, dtype=np.float32)

    # Init motor targets
    self.max_motor_speed = max_motor_speed
    self.motor_targets = self._default_angles
    self.prev_motor_targets = self._default_angles

    # Time management
    self._counter = 0
    self._n_substeps = n_substeps
    self._ctrl_dt = ctrl_dt

    # Phase init -> in real case self._ctrl_dt = self._n_substeps * self._sim_dt
    self._phase = np.array([0.0, np.pi])
    self._gait_freq = gait_freq
    self._phase_dt = 2 * np.pi * self._gait_freq * self._ctrl_dt 

    # Init joystick
    self._joystick = Gamepad(
        vel_range_x=vel_range_x,
        vel_range_y=vel_range_y,
        vel_range_rot=vel_range_rot,
        deadzone=0.03,
    )

  def get_obs(self, model, data) -> np.ndarray:
    # get gyro
    gyro = data.sensor("gyro").data
    # get accelerometer
    accelerometer = data.sensor("accelerometer").data
    # get gravity
    imu_xmat = data.site_xmat[model.site("imu").id].reshape(3, 3)
    gravity = imu_xmat.T @ np.array([0, 0, -1])
    # get joint angles delta and velocities
    joint_angles = data.qpos[7:] - self._default_angles
    joint_velocities = data.qvel[6:]
    # get command
    command = self._joystick.get_command()
    print(f"command: {command}")
    # adjust phase
    ph = self._phase if np.linalg.norm(command) >= 0.01 else np.ones(2) * np.pi
    phase = np.concatenate([np.cos(ph), np.sin(ph)])
    # concatenate all
    obs = np.hstack([
        gyro,
        accelerometer,
        gravity,
        command,
        joint_angles,
        joint_velocities,
        self._last_action,
        self._last_last_action,
        self._last_last_last_action,
        phase,
    ])
    return obs.astype(np.float32)

  def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    self._counter += 1
    if self._counter % self._n_substeps == 0:
      # obtain observation
      obs = self.get_obs(model, data)
      # run policy
      onnx_input = {"obs": obs.reshape(1, -1)}
      onnx_pred = self._policy.run(self._output_names, onnx_input)[0][0]
      # update action memory
      self._last_last_last_action = self._last_last_action.copy()
      self._last_last_action = self._last_action.copy()
      self._last_action = onnx_pred.copy()
      # update motor targets -> in real case self._ctrl_dt = self._n_substeps * self._sim_dt
      self.motor_targets = onnx_pred * self._action_scale + self._default_angles
      self.motor_targets = np.clip(self.motor_targets, 
                                self.prev_motor_targets - self.max_motor_speed * (self._ctrl_dt),
                                self.prev_motor_targets + self.max_motor_speed * (self._ctrl_dt)
                                )
      self.prev_motor_targets = self.motor_targets.copy()
      # apply control
      data.ctrl[:] = self.motor_targets
      # update phase 
      phase_tp1 = self._phase + self._phase_dt
      self._phase = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi


def load_callback(model=None, data=None):
  mujoco.set_mjcb_control(None)

  model = mujoco.MjModel.from_xml_path(
      bd5_constants.FEET_ONLY_FLAT_TERRAIN_XML.as_posix(),
      assets=get_assets(),
  )
  data = mujoco.MjData(model)

  mujoco.mj_resetDataKeyframe(model, data, 0)

  ctrl_dt = 0.02
  sim_dt = 0.002 # was 0.002
  n_substeps = int(round(ctrl_dt / sim_dt))
  model.opt.timestep = sim_dt

  policy = OnnxController(
      policy_path=(_ONNX_DIR / "bd5_test_policy.onnx").as_posix(),
      default_angles=np.array(model.keyframe("init_pose").qpos[7:]),
      ctrl_dt=ctrl_dt,
      n_substeps=n_substeps,
      action_scale=0.3,
      vel_range_x=[-0.6, 0.6],
      vel_range_y=[-0.6, 0.6],
      vel_range_rot=[-1.0, 1.0],
      gait_freq=1.0,
      max_motor_speed=4.82,
  )

  mujoco.set_mjcb_control(policy.get_control)

  return model, data

if __name__ == "__main__":
  viewer.launch(loader=load_callback)
