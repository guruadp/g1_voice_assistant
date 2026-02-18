#!/usr/bin/env python3
"""
Subtle custom arm motion test for G1 (variant: bent elbows) using low-level arm SDK channel.
Initialization and release are both smoothed.

Usage:
  python3 tests/test_arm_explain_motion_bent_elbows_sdk.py --net-if enP8p1s0 --duration 12
"""

import argparse
import math
import signal
import sys
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread


class G1JointIndex:
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftElbow = 18

    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightElbow = 25

    WaistYaw = 12
    kNotUsedJoint = 29


class BentElbowExplainMotion:
    def __init__(self, duration: float, amp: float, control_dt: float = 0.02, release_duration: float = 1.5):
        self.duration = duration
        self.amp = amp
        self.control_dt = control_dt

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.first_update_low_state = False
        self.crc = CRC()

        self._time = 0.0
        self._done = False
        self._released = False
        self.stop_requested = False

        self.init_duration = 2.0
        self.release_duration = release_duration

        self.kp = 55.0
        self.kd = 1.4

        self.joints = [
            G1JointIndex.LeftShoulderPitch,
            G1JointIndex.LeftShoulderRoll,
            G1JointIndex.LeftElbow,
            G1JointIndex.RightShoulderPitch,
            G1JointIndex.RightShoulderRoll,
            G1JointIndex.RightElbow,
            G1JointIndex.WaistYaw,
        ]

        # More bent elbows than the previous script.
        self.base = {
            G1JointIndex.LeftShoulderPitch: 0.04,
            G1JointIndex.LeftShoulderRoll: 0.12,
            G1JointIndex.LeftElbow: -0.90,
            G1JointIndex.RightShoulderPitch: 0.03,
            G1JointIndex.RightShoulderRoll: -0.10,
            G1JointIndex.RightElbow: -0.88,
            G1JointIndex.WaistYaw: 0.0,
        }

        self.last_targets = dict(self.base)
        self.start_pose = {}

    def init_channels(self):
        self.pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.pub.Init()

        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self.low_state_handler, 10)

    def low_state_handler(self, msg: LowState_):
        self.low_state = msg
        if not self.first_update_low_state:
            self.first_update_low_state = True

    def start(self):
        print("Waiting for lowstate...")
        while not self.first_update_low_state:
            time.sleep(0.1)

        for j in self.joints:
            self.start_pose[j] = self.low_state.motor_state[j].q

        self.ctrl_thread = RecurrentThread(
            interval=self.control_dt,
            target=self.write_cmd,
            name="g1_explain_motion_bent_elbows",
        )
        self.ctrl_thread.Start()

    def write_cmd(self):
        self._time += self.control_dt
        t = self._time
        if self.stop_requested and t < self.duration:
            self.duration = t
        blend_out = 2.0

        if t <= self.duration:
            self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1.0

            if t < self.init_duration:
                osc_gain = 0.0
            elif t > max(self.duration - blend_out, self.init_duration):
                osc_gain = max((self.duration - t) / blend_out, 0.0)
            else:
                osc_gain = 1.0

            init_ratio = np.clip(t / self.init_duration, 0.0, 1.0)

            w = 2.0 * math.pi * 0.42
            ls_pitch = self.base[G1JointIndex.LeftShoulderPitch] + osc_gain * (self.amp * 0.55) * math.sin(w * t)
            ls_roll = self.base[G1JointIndex.LeftShoulderRoll] + osc_gain * (self.amp * 0.65) * math.sin(w * t + 0.8)
            le = self.base[G1JointIndex.LeftElbow] + osc_gain * (self.amp * 0.10) * math.sin(w * t + 1.1)

            rs_pitch = self.base[G1JointIndex.RightShoulderPitch] + osc_gain * (self.amp * 0.50) * math.sin(w * t + 0.5)
            rs_roll = self.base[G1JointIndex.RightShoulderRoll] + osc_gain * (self.amp * 0.62) * math.sin(w * t + 1.3)
            re = self.base[G1JointIndex.RightElbow] + osc_gain * (self.amp * 0.10) * math.sin(w * t + 1.6)

            waist = self.base[G1JointIndex.WaistYaw] + osc_gain * (self.amp * 0.22) * math.sin(w * t + 0.2)

            targets = {
                G1JointIndex.LeftShoulderPitch: ls_pitch,
                G1JointIndex.LeftShoulderRoll: ls_roll,
                G1JointIndex.LeftElbow: le,
                G1JointIndex.RightShoulderPitch: rs_pitch,
                G1JointIndex.RightShoulderRoll: rs_roll,
                G1JointIndex.RightElbow: re,
                G1JointIndex.WaistYaw: waist,
            }

            for j in self.joints:
                q_start = self.start_pose.get(j, self.low_state.motor_state[j].q)
                q_cmd = (1.0 - init_ratio) * q_start + init_ratio * targets[j]
                self.low_cmd.motor_cmd[j].tau = 0.0
                self.low_cmd.motor_cmd[j].q = q_cmd
                self.low_cmd.motor_cmd[j].dq = 0.0
                self.low_cmd.motor_cmd[j].kp = self.kp
                self.low_cmd.motor_cmd[j].kd = self.kd
                self.last_targets[j] = q_cmd

        elif t <= self.duration + self.release_duration:
            r = np.clip((t - self.duration) / self.release_duration, 0.0, 1.0)
            # Smoothstep easing for softer handoff.
            e = r * r * (3.0 - 2.0 * r)
            self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1.0 - e

            for j in self.joints:
                q_now = self.low_state.motor_state[j].q
                q_cmd = (1.0 - e) * self.last_targets[j] + e * q_now
                self.low_cmd.motor_cmd[j].tau = 0.0
                self.low_cmd.motor_cmd[j].q = q_cmd
                self.low_cmd.motor_cmd[j].dq = 0.0
                # Extra damping near end to avoid snap.
                gain_scale = (1.0 - e) * (1.0 - e)
                self.low_cmd.motor_cmd[j].kp = gain_scale * self.kp
                self.low_cmd.motor_cmd[j].kd = gain_scale * self.kd

        elif not self._released:
            self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 0.0
            self._released = True
            self._done = True

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    def request_stop(self) -> None:
        self.stop_requested = True

    @property
    def done(self) -> bool:
        return self._done


def main():
    parser = argparse.ArgumentParser(description="Subtle G1 explaining arm motion (bent elbows variant)")
    parser.add_argument("--net-if", default="enP8p1s0", help="Robot network interface")
    parser.add_argument("--duration", type=float, default=12.0, help="Motion duration in seconds")
    parser.add_argument("--amp", type=float, default=0.10, help="Motion amplitude in radians")
    parser.add_argument("--release-duration", type=float, default=1.5, help="Release ramp duration in seconds")
    parser.add_argument("--no-prompt", action="store_true", help="Skip Enter confirmation prompt")
    args = parser.parse_args()

    print("WARNING: Ensure clear space around the robot before motion test.")
    if not args.no_prompt:
        input("Press Enter to continue...")

    ChannelFactoryInitialize(0, args.net_if)

    motion = BentElbowExplainMotion(
        duration=max(args.duration, 0.2),
        amp=float(np.clip(args.amp, 0.03, 0.20)),
        release_duration=max(0.05, args.release_duration),
    )
    motion.init_channels()
    motion.start()

    def _graceful_stop(_sig, _frame):
        motion.request_stop()

    signal.signal(signal.SIGINT, _graceful_stop)
    signal.signal(signal.SIGTERM, _graceful_stop)

    print("Running bent-elbow explaining motion...")
    while not motion.done:
        time.sleep(0.2)

    print("Done. arm_sdk released.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
