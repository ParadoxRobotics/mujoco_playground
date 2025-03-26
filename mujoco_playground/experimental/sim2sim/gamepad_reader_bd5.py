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
# pylint: disable=line-too-long
"""Logitech F710 Gamepad class that uses HID under the hood.

Adapted from motion_imitation: https://github.com/erwincoumans/motion_imitation/tree/master/motion_imitation/robots/gamepad/gamepad_reader.py.
"""
import threading
import time

import hid
import numpy as np

def _interpolate(value, range_value, deadzone=0.0):
    if value > 0.0:
      ret = value * range_value[1]
    else:
      ret = -value * range_value[0]
    if abs(ret) < deadzone:
      return 0.0
    return ret

class Gamepad:
  """Gamepad class that reads from a Logitech F710 gamepad."""

  def __init__(
      self,
      vendor_id=0x054c,
      product_id=0x05c4,
      vel_range_x=[-1.0, 1.0],
      vel_range_y=[-1.0, 1.0],
      vel_range_rot=[-1.0, 1.0],
      deadzone=0.01,
  ):
    self._vendor_id = vendor_id
    self._product_id = product_id
    self._vel_range_x = vel_range_x
    self._vel_range_y = vel_range_y
    self._vel_range_rot = vel_range_rot
    self._deadzone = deadzone

    self.vx = 0.0
    self.vy = 0.0
    self.wz = 0.0
    self.is_running = True

    self._device = None

    self.read_thread = threading.Thread(target=self.read_loop, daemon=True)
    self.read_thread.start()

  def _connect_device(self):
    try:
      self._device = hid.device()
      self._device.open(self._vendor_id, self._product_id)
      self._device.set_nonblocking(True)
      print(
          "Connected to"
          f" {self._device.get_manufacturer_string()} "
          f"{self._device.get_product_string()}"
      )
      return True
    except OSError as e:
      print(f"Error connecting to device: {e}")
      return False

  def read_loop(self):
    if not self._connect_device():
      self.is_running = False
      return

    while self.is_running:
      try:
        data = self._device.read(64)
        if data:
          self.update_command(data)
      except OSError as e:
        print(f"Error reading from device: {e}")

    self._device.close()

  def update_command(self, data):
    left_x = -(data[1] - 128) / 128.0
    left_y = -(data[2] - 128) / 128.0
    right_x = -(data[3] - 128) / 128.0

    self.vx = _interpolate(left_y, self._vel_range_x,  self._deadzone)
    self.vy = _interpolate(left_x, self._vel_range_y, self._deadzone)
    self.wz = _interpolate(right_x, self._vel_range_rot, self._deadzone)

  def get_command(self):
    return np.array([self.vx, self.vy, self.wz])

  def stop(self):
    self.is_running = False


if __name__ == "__main__":
  gamepad = Gamepad()
  while True:
    print(gamepad.get_command())
    time.sleep(0.1)
