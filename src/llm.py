import re

from openai import OpenAI

from nlu import action_style_hint


def ask_chat(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_text: str,
    requested_action: str | None = None,
    requested_locomotion: bool = False,
    history: list[dict] | None = None,
    enable_web_search: bool = False,
) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    if requested_action:
        messages.append({
            "role": "system",
            "content": (
                f"The user requested this supported action: {requested_action}. "
                f"Style guidance: {action_style_hint(requested_action)} "
                "Reply in 1-2 short spoken sentences. Sound warm and human, but do not use emoji. "
                "Do not explicitly narrate the action execution, and avoid lines like 'I am doing a handshake now'."
            ),
        })
    if requested_locomotion:
        messages.append({
            "role": "system",
            "content": (
                "The user asked for locomotion. Confirm naturally that you can do the movement. "
                "Do not say you cannot move. Keep it short and human-like."
            ),
        })
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    if enable_web_search:
        try:
            resp = client.responses.create(
                model=model,
                input=messages,
                tools=[{"type": "web_search_preview"}],
                temperature=0.4,
            )
            text = getattr(resp, "output_text", "") or ""
            if text.strip():
                return text.strip()
        except Exception as exc:
            print(f"[web] search unavailable, fallback to chat: {exc}")

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.4,
    )
    return (resp.choices[0].message.content or "").strip()


def compact_spoken_reply(text: str, max_sentences: int = 2) -> str:
    if not text:
        return ""
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("* ") or line.startswith("- "):
            continue
        lines.append(line)
    cleaned = " ".join(lines)
    cleaned = re.sub(r"\[[^\]]+\]\([^)]+\)", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""

    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    out = " ".join(parts[:max_sentences]).strip()
    return out or cleaned
