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

from scipy.spatial.transform import Rotation as R

from mujoco_playground._src.locomotion.bd5 import bd5_constants
from mujoco_playground._src.locomotion.bd5.base import get_assets
from mujoco_playground.experimental.sim2sim.gamepad_reader_bd5 import Gamepad

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE / "onnx"

class OnnxController:
  """ONNX controller for the BD-5."""

  def __init__(
      self,
      data_path: str,
      policy_path: str,
      default_angles: np.ndarray,
      ctrl_dt: float,
      n_substeps: int,
      action_scale: float = 0.3,
      max_motor_speed: float = 4.82,
      action_filter: bool = False,
  ):
    self._output_names = ["continuous_actions"]
    self._policy = rt.InferenceSession(
        policy_path, providers=["CPUExecutionProvider"]
    )

    # Init action scale and memory
    self._action_scale = action_scale
    self._default_angles = default_angles

    # Init motor targets
    self.max_motor_speed = max_motor_speed
    self.motor_targets = self._default_angles
    self.prev_motor_targets = self._default_angles
    self.exp_filter = action_filter
    self.prev_filter_state = self._default_angles

    # Time management
    self._counter = 0
    self._n_substeps = n_substeps
    self._ctrl_dt = ctrl_dt

    self.data_log = np.load(data_path)
    self.nb_data_log = self.data_log.shape[0] - 1
    self.log_step = 0
    self.error = []

  def get_obs(self, model, data) -> np.ndarray:
    # get data from file :
    if self.log_step < self.nb_data_log:
        obs = self.data_log[self.log_step, :]
        self.log_step += 1
    else:
      obs = np.zeros_like(self.data_log[0, :])
    return obs.astype(np.float32)

  def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    self._counter += 1
    if self._counter % self._n_substeps == 0:
      # obtain observation
      obs = self.get_obs(model, data)
      # run policy
      onnx_input = {"obs": obs.reshape(1, -1)}
      onnx_pred = self._policy.run(self._output_names, onnx_input)[0][0]
      # update motor targets -> in real case self._ctrl_dt = self._n_substeps * self._sim_dt
      self.motor_targets = onnx_pred * self._action_scale + self._default_angles
      # filter action 
      if self.exp_filter:
        filter_state = 0.8 * self.prev_filter_state + 0.2 * self.motor_targets
        self.prev_filter_state = filter_state.copy()
        self.motor_targets = filter_state
      # clip speed 
      if self.max_motor_speed is not None:
        self.motor_targets = np.clip(self.motor_targets, 
                                  self.prev_motor_targets - self.max_motor_speed * (self._ctrl_dt),
                                  self.prev_motor_targets + self.max_motor_speed * (self._ctrl_dt)
                                  )
      self.prev_motor_targets = self.motor_targets.copy() 
      # apply control
      data.ctrl[:] = self.motor_targets


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
      data_path="/home/master/mujoco_playground/mujoco_playground/experimental/sim2sim/bd5_state.npy",
      policy_path=(_ONNX_DIR / "test.onnx").as_posix(),
      default_angles=np.array(model.keyframe("init_pose").qpos[7:]),
      ctrl_dt=ctrl_dt,
      n_substeps=n_substeps,
      action_scale=0.3,
      max_motor_speed=4.50,
      action_filter=False
  )

  mujoco.set_mjcb_control(policy.get_control)

  return model, data

if __name__ == "__main__":
  viewer.launch(loader=load_callback)
