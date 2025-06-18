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

# Domain randomization for the BD5 environment
# See Booster T1 in the locomotion folder for more details
# And DeepMind soccer humanoid for value range

import jax
import jax.numpy as jp
from mujoco import mjx

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1
IMU_SITE_ID = 0

def quat_mul(q1: jax.Array, q2: jax.Array) -> jax.Array:
    """Multiplies two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return jp.array([w, x, y, z])

def euler_to_quat(euler: jax.Array) -> jax.Array:
    """Converts Euler angles (roll, pitch, yaw) to a quaternion."""
    roll, pitch, yaw = euler
    cy = jp.cos(yaw * 0.5)
    sy = jp.sin(yaw * 0.5)
    cp = jp.cos(pitch * 0.5)
    sp = jp.sin(pitch * 0.5)
    cr = jp.cos(roll * 0.5)
    sr = jp.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return jp.array([w, x, y, z])

def domain_randomize(model: mjx.Model, rng: jax.Array):
    @jax.vmap
    def rand_dynamics(rng):
        # Floor friction: =U(0.5, 1.0).
        rng, key = jax.random.split(rng)
        geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
            jax.random.uniform(key, minval=0.5, maxval=1.0)
        )

        # Joint frictionloss: *U(0.9, 1.1).
        rng, key = jax.random.split(rng)
        frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(
            key, shape=(10,), minval=0.9, maxval=1.1
        )
        dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)

        # Scale armature: *U(1.0, 1.05).
        rng, key = jax.random.split(rng)
        armature = model.dof_armature[6:] * jax.random.uniform(
            key, shape=(10,), minval=1.0, maxval=1.05
        )
        dof_armature = model.dof_armature.at[6:].set(armature)

        # Scale all link masses: *U(0.9, 1.1).
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(
            key, shape=(model.nbody,), minval=0.9, maxval=1.1
        )
        body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

        # Add mass to torso: +U(-0.2, 0.2).
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(key, minval=-0.2, maxval=0.2)
        body_mass = body_mass.at[TORSO_BODY_ID].set(
            body_mass[TORSO_BODY_ID] + dmass
        )

        # Jitter center of mass position: +U(-0.05, 0.05).
        rng, key = jax.random.split(rng)
        dpos = jax.random.uniform(key, (3,), minval=-0.05, maxval=0.05) 
        body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(
            model.body_ipos[TORSO_BODY_ID] + dpos
        )

        # Jitter qpos0: +U(-0.05, 0.05).
        rng, key = jax.random.split(rng)
        qpos0 = model.qpos0
        qpos0 = qpos0.at[7:].set(
            qpos0[7:] + jax.random.uniform(key, shape=(10,), minval=-0.05, maxval=0.05)
        )

        # Joint stiffness: *U(0.9, 1.1).
        rng, key = jax.random.split(rng)
        kp = model.actuator_gainprm[:, 0] * jax.random.uniform(
            key, (10,), minval=0.9, maxval=1.1
        )
        actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp)
        actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp)

        # IMU site position: +U(-0.005, 0.005)
        rng, key = jax.random.split(rng)
        dpos_imu = jax.random.uniform(key, (3,), minval=-0.005, maxval=0.005) 
        # Add the jitter to the original IMU site position
        site_pos = model.site_pos.at[IMU_SITE_ID].set(
            model.site_pos[IMU_SITE_ID] + dpos_imu
        )

        # IMU site orientation: +U(-0.0524, 0.0524)
        rng, key = jax.random.split(rng)
        deuler_imu = jax.random.uniform(key, (3,), minval=-0.0524, maxval=0.0524)
        dquat_imu = euler_to_quat(deuler_imu)
        # Multiply the original IMU site quaternion by the jitter quaternion
        # Note: MuJoCo sites don't have an orientation by default unless you specify one.
        # If model.site_quat[imu_site_id] is [1,0,0,0] (no rotation), this just sets it to dquat_imu.
        # If you ever add a base rotation to your IMU site in the XML, this will correctly compound it.
        site_quat = model.site_quat.at[IMU_SITE_ID].set(
            quat_mul(model.site_quat[IMU_SITE_ID], dquat_imu)
        )

        return (
            geom_friction,
            dof_frictionloss,
            dof_armature,
            body_ipos,
            body_mass,
            qpos0,
            actuator_gainprm,
            actuator_biasprm,
            site_pos,
            site_quat
        )

    (
        friction,
        frictionloss,
        armature,
        body_ipos,
        body_mass,
        qpos0,
        actuator_gainprm,
        actuator_biasprm,
        site_pos,
        site_quat
    ) = rand_dynamics(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({
        "geom_friction": 0,
        "dof_frictionloss": 0,
        "dof_armature": 0,
        "body_ipos" : 0,
        "body_mass": 0,
        "qpos0": 0,
        "actuator_gainprm": 0,
        "actuator_biasprm": 0,
        "site_pos": 0,
        "site_quat": 0
    })

    model = model.tree_replace({
        "geom_friction": friction,
        "dof_frictionloss": frictionloss,
        "dof_armature": armature,
        "body_ipos": body_ipos,
        "body_mass": body_mass,
        "qpos0": qpos0,
        "actuator_gainprm": actuator_gainprm,
        "actuator_biasprm": actuator_biasprm,
        "site_pos": site_pos,
        "site_quat": site_quat
    })

    return model, in_axes