from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

from nlu import (
    contains_action_hint,
    contains_loco_hint,
    has_negation,
    parse_loco_command,
    split_requests,
)


def apply_loco_commands(text: str, loco_client: LocoClient | None, args, dry_run: bool = False) -> bool:
    executed_any = False
    for chunk in split_requests(text):
        if has_negation(chunk) and (contains_action_hint(chunk) or contains_loco_hint(chunk)):
            print(f"Skipped (negated): {chunk}")
            continue

        loco_cmd = parse_loco_command(
            chunk,
            walk_speed=args.walk_speed,
            lateral_speed=args.lateral_speed,
            turn_speed=args.turn_speed,
            default_duration=args.default_duration,
            seconds_per_step=args.seconds_per_step,
        )
        if loco_cmd is None:
            continue

        vx = float(loco_cmd["vx"])
        vy = float(loco_cmd["vy"])
        omega = float(loco_cmd["omega"])
        duration = float(loco_cmd["duration"])

        if dry_run:
            print(f"[DRY-RUN] Would move: vx={vx:.2f}, vy={vy:.2f}, omega={omega:.2f}, duration={duration:.2f}s")
            executed_any = True
            continue

        if loco_client is None:
            print("[loco] client unavailable; skipping move command")
            continue

        code = loco_client.SetVelocity(vx, vy, omega, duration)
        if code != 0:
            print(f"Loco failed. code={code}, vx={vx}, vy={vy}, omega={omega}, duration={duration}")
            continue

        executed_any = True
        print(f"Moved: vx={vx:.2f}, vy={vy:.2f}, omega={omega:.2f}, duration={duration:.2f}s")
    return executed_any
