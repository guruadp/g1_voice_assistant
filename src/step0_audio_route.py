import argparse
import os
import subprocess
import sys


def run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip() or f"command failed: {' '.join(cmd)}"
        raise RuntimeError(msg)
    return proc.stdout


def list_short(kind: str) -> list[str]:
    out = run(["pactl", "list", "short", kind])
    lines = [ln for ln in out.splitlines() if ln.strip()]
    return lines


def get_info() -> str:
    return run(["pactl", "info"])


def parse_default(info: str, key: str) -> str:
    for line in info.splitlines():
        if line.startswith(key):
            return line.split(":", 1)[1].strip()
    return ""


def set_default(sink: str | None, source: str | None) -> None:
    if sink:
        run(["pactl", "set-default-sink", sink])
    if source:
        run(["pactl", "set-default-source", source])


def auto_pick_anker(sinks: list[str], sources: list[str]) -> tuple[str | None, str | None]:
    sink_name = None
    source_name = None

    for ln in sinks:
        cols = ln.split("\t")
        if len(cols) >= 2 and "Anker_PowerConf" in cols[1]:
            sink_name = cols[1]
            break

    for ln in sources:
        cols = ln.split("\t")
        if len(cols) >= 2 and "Anker_PowerConf" in cols[1]:
            source_name = cols[1]
            break

    return sink_name, source_name


def print_state() -> None:
    info = get_info()
    default_sink = parse_default(info, "Default Sink")
    default_source = parse_default(info, "Default Source")

    print(f"Default Sink:   {default_sink}")
    print(f"Default Source: {default_source}")
    print("\nAvailable sinks:")
    for ln in list_short("sinks"):
        print(f"  {ln}")
    print("\nAvailable sources:")
    for ln in list_short("sources"):
        print(f"  {ln}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 0: configure PulseAudio sink/source for this project")
    parser.add_argument("--sink", help="Set default sink name")
    parser.add_argument("--source", help="Set default source name")
    parser.add_argument("--auto-anker", action="store_true", help="Auto-select first sink/source containing Anker_PowerConf")
    parser.add_argument("--set-pulse-runtime", action="store_true", help="Set PULSE_RUNTIME_PATH=/run/user/<uid>/pulse/")
    args = parser.parse_args()

    try:
        if args.set_pulse_runtime:
            os.environ["PULSE_RUNTIME_PATH"] = f"/run/user/{os.getuid()}/pulse/"
            print(f"PULSE_RUNTIME_PATH={os.environ['PULSE_RUNTIME_PATH']}")

        sinks = list_short("sinks")
        sources = list_short("sources")

        sink = args.sink
        source = args.source

        if args.auto_anker:
            auto_sink, auto_source = auto_pick_anker(sinks, sources)
            sink = sink or auto_sink
            source = source or auto_source

        if sink or source:
            set_default(sink=sink, source=source)
            print("Updated defaults.")

        print_state()

    except RuntimeError as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
