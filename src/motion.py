import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient, action_map

CUSTOM_MOTION_SCRIPTS = {
    "custom:explain_basic": "arm_explain_motion_basic_sdk.py",
    "custom:explain_bent_elbows": "arm_explain_motion_bent_elbows_sdk.py",
    "custom:explain_open_palms": "arm_explain_motion_open_palms_sdk.py",
}


class ArmGestureController:
    def __init__(self, net_if: str, dry_run: bool = False):
        self.enabled = False
        self.net_if = net_if
        self.dry_run = dry_run
        self._client = None
        if self.dry_run:
            self.enabled = True
            print("[DRY-RUN] Motion controller active (no arm commands will be sent).")
            return
        try:
            ChannelFactoryInitialize(0, net_if)
            self._client = G1ArmActionClient()
            self._client.SetTimeout(10.0)
            self._client.Init()
            self.enabled = True
            print(f"[motion] SDK ready on net_if={net_if}")
        except Exception as exc:
            print(f"[motion] SDK init failed: {exc}")

    def execute(self, style: str, auto_release: bool = True) -> None:
        if self.dry_run:
            print(f"[DRY-RUN] Would execute arm action: {style}")
            if auto_release:
                print("[DRY-RUN] Would release arm")
            return
        if not self.enabled:
            print(f"[motion] skipped (SDK not ready): {style}")
            return

        action_id = action_map.get(style)
        if action_id is None:
            print(f"[motion] unknown action style: {style}")
            return

        self._client.ExecuteAction(action_id)
        time.sleep(0.45)
        if auto_release:
            self.release_arm()

    def release_arm(self) -> None:
        if self.dry_run:
            print("[DRY-RUN] Would release arm")
            return
        if not self.enabled:
            return
        release_id = action_map.get("release arm")
        if release_id is None:
            return
        self._client.ExecuteAction(release_id)
        time.sleep(0.35)

    def launch_custom_process(self, style: str, duration_s: float = 20.0, release_duration_s: float = 0.2):
        if self.dry_run:
            print(f"[DRY-RUN] Would launch custom motion {style} for {duration_s:.2f}s")
            return None
        script_name = CUSTOM_MOTION_SCRIPTS.get(style)
        if not script_name:
            print(f"[motion] unknown custom motion: {style}")
            return None

        script_path = Path(__file__).resolve().parent / script_name
        if not script_path.exists():
            print(f"[motion] custom motion script not found: {script_path}")
            return None

        cmd = [
            sys.executable,
            str(script_path),
            "--net-if",
            self.net_if,
            "--duration",
            str(max(0.2, float(duration_s))),
            "--amp",
            "0.10",
            "--release-duration",
            str(max(0.05, float(release_duration_s))),
            "--no-prompt",
        ]
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop_custom_process(self, proc, timeout_s: float = 2.5) -> bool:
        if proc is None:
            return True
        if proc.poll() is not None:
            return True

        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=timeout_s)
            return True
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=0.6)
                return False
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=1.0)
                return False


def start_motion_thread(motion: ArmGestureController, gesture: str, auto_release: bool = True) -> threading.Thread:
    def _run_motion() -> None:
        try:
            motion.execute(gesture, auto_release=auto_release)
        except Exception as exc:
            print(f"[motion] execution error: {exc}")

    t = threading.Thread(target=_run_motion, daemon=True)
    t.start()
    return t
