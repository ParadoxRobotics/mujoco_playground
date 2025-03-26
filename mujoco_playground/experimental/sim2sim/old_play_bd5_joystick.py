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
from mujoco_playground.experimental.sim2sim.gamepad_reader import Gamepad

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
      action_scale: float = 0.5,
      vel_scale_x: float = 1.0,
      vel_scale_y: float = 1.0,
      vel_scale_rot: float = 1.0,
      gait_freq:float = 1.5,
  ):
    self._output_names = ["continuous_actions"]
    self._policy = rt.InferenceSession(
        policy_path, providers=["CPUExecutionProvider"]
    )

    self._action_scale = action_scale
    self._default_angles = default_angles
    self._last_action = np.zeros_like(default_angles, dtype=np.float32)
    self._last_last_action = np.zeros_like(default_angles, dtype=np.float32)
    self._last_last_last_action = np.zeros_like(default_angles, dtype=np.float32)

    self._counter = 0
    self._n_substeps = n_substeps
    self._ctrl_dt = ctrl_dt

    self._phase = np.array([0.0, np.pi])
    self._gait_freq = gait_freq
    self._phase_dt = 2 * np.pi * self._gait_freq * ctrl_dt

    self._joystick = Gamepad(
        vel_scale_x=vel_scale_x,
        vel_scale_y=vel_scale_y,
        vel_scale_rot=vel_scale_rot,
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
    # get joint angles and velocities
    joint_angles = data.qpos[7:] - self._default_angles
    joint_velocities = data.qvel[6:]
    # get command
    command = self._joystick.get_command()
    # adjust phase
    ph = self._phase if np.linalg.norm(command) >= 0.01 else np.ones(2) * np.pi
    phase = np.concatenate([np.cos(ph), np.sin(ph)])
    joint_angles[:2] *= 0.0
    joint_velocities[:2] *= 0.0
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
      # apply control
      data.ctrl[:] = onnx_pred * self._action_scale + self._default_angles
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
  sim_dt = 0.004 # was 0.002
  n_substeps = int(round(ctrl_dt / sim_dt))
  model.opt.timestep = sim_dt

  policy = OnnxController(
      policy_path=(_ONNX_DIR / "bd5_policy.onnx").as_posix(),
      default_angles=np.array(model.keyframe("init_pose").qpos[7:]),
      ctrl_dt=ctrl_dt,
      n_substeps=n_substeps,
      action_scale=0.3,
      vel_scale_x=1.0,
      vel_scale_y=0.8,
      vel_scale_rot=0.7,
      gait_freq=1.5,
  )

  mujoco.set_mjcb_control(policy.get_control)

  return model, data

if __name__ == "__main__":
  viewer.launch(loader=load_callback)
