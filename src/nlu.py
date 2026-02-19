import re
from dataclasses import dataclass

EXPLAIN_CUSTOM_ACTIONS = [
    "custom:explain_basic",
    "custom:explain_bent_elbows",
    "custom:explain_open_palms",
]

NEGATION_WORDS = {"not", "dont", "don't", "do not", "never", "no"}
SPLIT_PATTERN = re.compile(r"\b(?:and then|then|after that|afterwards|and)\b|[,;]+")

ACTION_SPEECH_STYLE = {
    "shake hand": "For handshake: invite the user to extend their hand and add a warm social line like 'nice to meet you'.",
    "hug": "For hug: invite the user to come a little closer, in a kind and respectful tone.",
    "high five": "For high five: use an upbeat line asking the user to raise their hand.",
    "high wave": "For high wave: use a cheerful greeting line.",
    "face wave": "For face wave: use a friendly polite line for thanks or hello.",
    "clap": "For clap: use a celebratory line.",
    "hands up": "For hands up: use a playful and safe instruction line.",
    "heart": "For heart: use a warm affectionate line.",
    "reject": "For reject: be polite, calm, and gentle in refusal wording.",
}


@dataclass
class Intent:
    speech_action: str | None
    motion_action: str | None
    loco_cmds: list[dict]
    wants_explain: bool


def map_action_by_text(text: str) -> str | None:
    t = text.lower()

    if any(k in t for k in ["handshake", "shake hand", "shake hands"]):
        return "shake hand"
    if any(k in t for k in ["high five", "hi five"]):
        return "high five"
    if any(k in t for k in ["face wave"]):
        return "face wave"
    if any(k in t for k in ["high wave", "wave high"]):
        return "high wave"
    if any(k in t for k in ["wave", "hello", "hi"]):
        return "high wave"
    if any(k in t for k in ["thank", "thanks"]):
        return "face wave"
    if any(k in t for k in ["hug"]):
        return "hug"
    if any(k in t for k in ["clap"]):
        return "clap"
    if any(k in t for k in ["hands up", "raise hands", "put your hands up"]):
        return "hands up"
    if any(k in t for k in ["heart", "love gesture"]):
        return "heart"
    if re.search(r"\b(reject|no|can\'t|cannot|refuse)\b", t):
        return "reject"
    if any(k in t for k in ["right hand up"]):
        return "right hand up"
    if any(k in t for k in ["x-ray", "xray"]):
        return "x-ray"
    if any(k in t for k in ["left kiss"]):
        return "left kiss"
    if any(k in t for k in ["right kiss"]):
        return "right kiss"
    if re.search(r"\bkiss\b", t):
        return "two-hand kiss"
    if any(k in t for k in ["two-hand kiss", "two hand kiss"]):
        return "two-hand kiss"
    return None


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("_", " ")
    text = text.replace("donot", "do not")
    text = re.sub(r"[^a-z0-9\-\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_requests(text: str):
    parts = [p.strip() for p in SPLIT_PATTERN.split(text) if p.strip()]
    return parts if parts else [text.strip()]


def has_negation(text: str) -> bool:
    t = normalize(text)
    if not t:
        return False
    return any(n in t for n in NEGATION_WORDS)


def contains_action_hint(text: str) -> bool:
    return map_action_by_text(text) is not None


def contains_loco_hint(text: str) -> bool:
    t = normalize(text)
    if not t:
        return False

    if any(k in t for k in ["stop", "halt", "freeze", "stop moving", "stop walk"]):
        return True

    has_turn_word = any(k in t for k in ["turn", "rotate", "spin", "yaw"])
    has_move_word = any(k in t for k in ["walk", "move", "go", "step"])
    has_direction_word = any(k in t for k in ["forward", "backward", "back", "lateral", "sideways"])

    if has_turn_word or has_move_word or has_direction_word:
        return True

    if ("left" in t or "right" in t) and (has_turn_word or has_move_word or "lateral" in t or "sideways" in t):
        return True

    return False


def parse_duration_seconds(text: str, default_duration: float, seconds_per_step: float) -> float:
    t = normalize(text)

    sec_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|sec|s)\b", t)
    if sec_match:
        return max(0.1, float(sec_match.group(1)))

    step_match = re.search(r"(\d+(?:\.\d+)?)\s*steps?\b", t)
    if step_match:
        steps = float(step_match.group(1))
        return max(0.1, steps * max(0.1, seconds_per_step))

    return max(0.1, default_duration)


def parse_loco_command(
    text: str,
    walk_speed: float,
    lateral_speed: float,
    turn_speed: float,
    default_duration: float,
    seconds_per_step: float,
):
    t = normalize(text)
    if not t:
        return None

    if any(k in t for k in ["stop", "halt", "freeze", "stop moving", "stop walk"]):
        return {"vx": 0.0, "vy": 0.0, "omega": 0.0, "duration": 0.2, "is_stop": 1.0}

    has_turn_word = any(k in t for k in ["turn", "rotate", "spin", "yaw"])
    has_move_word = any(k in t for k in ["walk", "move", "go", "step"])

    vx = 0.0
    vy = 0.0
    omega = 0.0

    if has_turn_word:
        if "left" in t and "right" not in t:
            omega = abs(turn_speed)
        elif "right" in t and "left" not in t:
            omega = -abs(turn_speed)
        elif "counterclockwise" in t or "anticlockwise" in t:
            omega = abs(turn_speed)
        elif "clockwise" in t:
            omega = -abs(turn_speed)
        else:
            omega = abs(turn_speed)

    if has_move_word or any(k in t for k in ["forward", "backward", "back", "lateral", "sideways"]):
        if "forward" in t:
            vx = abs(walk_speed)
        elif "backward" in t or re.search(r"\bback\b", t):
            vx = -abs(walk_speed)

        lateral_intent = any(k in t for k in ["lateral", "sideways", "strafe"])
        if lateral_intent or (("left" in t or "right" in t) and not has_turn_word):
            if "left" in t and "right" not in t:
                vy = abs(lateral_speed)
            elif "right" in t and "left" not in t:
                vy = -abs(lateral_speed)

    if vx == 0.0 and vy == 0.0 and omega == 0.0 and has_move_word:
        if any(k in t for k in ["walk", "go", "step"]):
            vx = abs(walk_speed)

    if vx == 0.0 and vy == 0.0 and omega == 0.0:
        return None

    duration = parse_duration_seconds(t, default_duration, seconds_per_step)
    return {"vx": vx, "vy": vy, "omega": omega, "duration": duration, "is_stop": 0.0}


def is_explicit_action_request(text: str, action: str | None) -> bool:
    if not action:
        return False
    t = text.lower()
    if action in {"high wave", "face wave"} and re.search(r"^\s*(hi|hello|hey)\b", t):
        if not re.search(r"\b(wave|gesture|do|perform|show|give)\b", t):
            return False

    return bool(
        re.search(r"\b(can you|could you|please|give me|do|perform|show|make|let's)\b", t)
        or "?" in t
    )


def is_explain_request(text: str) -> bool:
    t = text.lower().strip()
    patterns = [
        r"\bexplain\b",
        r"\bcan you explain\b",
        r"\bwhat is\b",
        r"\bhow does\b",
        r"\bhow do\b",
        r"\bwhy\b",
        r"\btell me about\b",
        r"\bhelp me understand\b",
    ]
    return any(re.search(p, t) for p in patterns)


def is_social_intro_request(text: str) -> bool:
    t = text.lower().strip()
    patterns = [
        r"\bwho are you\b",
        r"\bwhat is your name\b",
        r"\byour name\b",
        r"\bintroduce yourself\b",
    ]
    return any(re.search(p, t) for p in patterns)


def choose_action(user_text: str, reply_text: str, explain_index: int) -> tuple[str | None, int]:
    action = map_action_by_text(user_text)
    if action:
        return action, explain_index

    if is_social_intro_request(user_text):
        return "high wave", explain_index

    if is_explain_request(user_text):
        custom = EXPLAIN_CUSTOM_ACTIONS[explain_index % len(EXPLAIN_CUSTOM_ACTIONS)]
        return custom, explain_index + 1

    custom = EXPLAIN_CUSTOM_ACTIONS[explain_index % len(EXPLAIN_CUSTOM_ACTIONS)]
    return custom, explain_index + 1


def action_phrase(action: str) -> str:
    phrase_map = {
        "shake hand": "a handshake",
        "high five": "a high five",
        "high wave": "a high wave",
        "face wave": "a face wave",
        "clap": "a clap",
        "hug": "a hug",
        "hands up": "hands up",
        "heart": "a heart gesture",
        "right hand up": "right hand up",
        "x-ray": "an x-ray pose",
        "left kiss": "a left kiss",
        "right kiss": "a right kiss",
        "two-hand kiss": "a two-hand kiss",
        "reject": "a reject gesture",
    }
    return phrase_map.get(action, action)


def action_style_hint(action: str | None) -> str:
    if not action:
        return ""
    return ACTION_SPEECH_STYLE.get(action, "Use a natural human-like line that matches the action.")


def parse_intent(
    text: str,
    explain_index: int,
    walk_speed: float,
    lateral_speed: float,
    turn_speed: float,
    default_duration: float,
    seconds_per_step: float,
) -> tuple[Intent, int]:
    requested_action = map_action_by_text(text)
    speech_action = requested_action if is_explicit_action_request(text, requested_action) else None
    wants_explain = is_explain_request(text)

    loco_cmds: list[dict] = []
    for chunk in split_requests(text):
        if has_negation(chunk) and (contains_action_hint(chunk) or contains_loco_hint(chunk)):
            continue
        loco_cmd = parse_loco_command(
            chunk,
            walk_speed=walk_speed,
            lateral_speed=lateral_speed,
            turn_speed=turn_speed,
            default_duration=default_duration,
            seconds_per_step=seconds_per_step,
        )
        if loco_cmd is not None:
            loco_cmds.append(loco_cmd)

    loco_only_request = bool(loco_cmds) and speech_action is None and not wants_explain
    if loco_only_request:
        motion_action = None
        next_explain_index = explain_index
    else:
        motion_action, next_explain_index = choose_action(text, "", explain_index)

    return Intent(
        speech_action=speech_action,
        motion_action=motion_action,
        loco_cmds=loco_cmds,
        wants_explain=wants_explain,
    ), next_explain_index
