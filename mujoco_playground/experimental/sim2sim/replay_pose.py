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
  ):

    self.data_log = np.load(data_path)
    self.nb_data_log = self.data_log.shape[0] - 1
    self.log_step = 0
    self.error = []

  def get_obs(self) -> np.ndarray:
    # get data from file :
    if self.log_step < self.nb_data_log:
        obs = self.data_log[self.log_step, :]
        self.log_step += 1
    else:
      obs = np.zeros_like(self.data_log[0, :])
    return obs.astype(np.float32)

  def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
      # obtain observation
      obs = self.get_obs()
      # apply control
      data.ctrl[:] = obs


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
  )

  mujoco.set_mjcb_control(policy.get_control)

  return model, data

if __name__ == "__main__":
  viewer.launch(loader=load_callback)
