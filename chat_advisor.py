"""
chat_advisor.py - the language model behind the passenger-plan chat.

server.py owns the forecast. This module takes the numbers the board is already
showing, adds whatever the rider says about their age, work, commute and trip
length, and asks a free OpenRouter model for a short travel answer. The key
stays on the server; the browser only ever sees the reply.

    OPENROUTER_API_KEY   required. Free key: https://openrouter.ai/keys
    OPENROUTER_MODEL     optional. Default: nvidia/nemotron-3-ultra-550b-a55b:free

No extra dependency: the call is a plain HTTPS POST through urllib.
"""
import json
import logging
import os
import urllib.error
import urllib.request

logger = logging.getLogger("chat_advisor")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
FALLBACK_MODELS = (
    "nvidia/nemotron-3.5-lightning:free",
    "google/gemma-4-31b-it:free",
    "openrouter/free",
)
MAX_HISTORY = 8
MAX_MESSAGE_CHARS = 800
MAX_REPLY_TOKENS = 400
TIMEOUT_SECONDS = 45


class ChatUnavailable(Exception):
    """The chat cannot answer; carries the HTTP status the API should return."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def api_key():
    return os.environ.get("OPENROUTER_API_KEY", "").strip()


def models():
    """The configured model first, then the free fallbacks that actually answer."""
    primary = os.environ.get("OPENROUTER_MODEL", "").strip() or DEFAULT_MODEL
    return [primary] + [name for name in FALLBACK_MODELS if name != primary]


def _num(context, key, suffix="", digits=0):
    value = (context or {}).get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return f"{number:.{digits}f}{suffix}"


def _line(label, value):
    return f"- {label}: {value}" if value else None


def system_prompt(context=None, profile="Healthy Adult"):
    """The board's numbers, written as things the model may not invent."""
    context = context or {}
    replay = context.get("is_replay") is True
    lines = [
        "You are the passenger-plan assistant on an Anand Vihar (Delhi) air-quality board.",
        "Answer only from the numbers below. Never invent an AQI, an hour, or a study.",
        "The board is one CPCB station, so say \"at Anand Vihar\" rather than \"in Delhi\".",
        "You are not a doctor: no diagnosis, no medication changes, no promises about health.",
        "",
        "THE BOARD RIGHT NOW",
        "- Station: DPCC Anand Vihar, Delhi (single station, "
        + ("recorded 2023 replay - NOT today's air" if replay else "live hourly sensor reading")
        + ")",
        _line("Reading", context.get("reading_time")),
        _line("Current AQI", " ".join(
            part for part in (_num(context, "current_aqi"), context.get("cur_cat")) if part
        )),
    ]

    reading = _num(context, "current_aqi")
    forecast = _num(context, "pred_24h")
    if forecast:
        head = f"- +24h forecast: {forecast} AQI"
        if reading and float(context.get("pred_24h")) > float(context.get("current_aqi")):
            head += " (worse than the reading)"
        elif reading:
            head += " (better than the reading)"
        lines.append(head)
    lines.append(_line("Worst stop on the line", _num(context, "peak_24h") + " AQI at " + str(context.get("worst_hour"))
                       if _num(context, "peak_24h") and context.get("worst_hour") else None))
    lines.append(_line("Cleanest stop on the line", _num(context, "min_24h") + " AQI at " + str(context.get("best_hour"))
                       if _num(context, "min_24h") and context.get("best_hour") else None))
    low, high = _num(context, "pi80_low"), _num(context, "pi80_high")
    if low and high:
        lines.append(f"- Honest +24h range (80%): {low}-{high} AQI")
    risk = _num(context, "severe_pct", "%", 0)
    if risk:
        lines.append(f"- Severe-day risk (AQI above 400 within 24 hours): {risk}")
    lines.append(_line("At the reading - PM2.5", _num(context, "pm25") and _num(context, "pm25") + " ug/m3"))
    lines.append(_line("At the reading - PM10", _num(context, "pm10") and _num(context, "pm10") + " ug/m3"))
    weather = [part for part in (
        _num(context, "temp", " C", 1),
        _num(context, "humidity", "% humidity"),
        _num(context, "wind", " km/h wind", 1),
    ) if part]
    if weather:
        lines.append("- Weather at the reading: " + ", ".join(weather))

    window = context.get("window") if isinstance(context.get("window"), dict) else {}
    if window.get("measured") is not None:
        cams = int(window.get("cams") or 0)
        lines.append(
            f"- Input window: {int(window.get('measured') or 0)} measured hours"
            + (f" + {cams} CAMS-modelled hours" if cams else "")
        )

    wave = [point for point in (context.get("wave") or []) if isinstance(point, dict)][:7]
    stops = [
        f"{point.get('time')} ({point.get('horizon')}): {point.get('aqi')}"
        for point in wave if point.get("aqi") is not None
    ]
    if stops:
        lines.append("- The line, stop by stop: " + "; ".join(stops))

    trip = _num(context, "trip_length_min", " minutes")
    if trip:
        lines.append(f"- Trip length selected on the plan: {trip}")
    lines.append(f"- Health profile selected on the board: {profile}")

    lines += [
        "",
        "HOW TO ANSWER",
        "- Under 120 words. Plain sentences, no headings, no tables, no emoji.",
        "- Use the rider's age, work, commute mode, route and trip length when they gave them, and name the hour to leave in IST.",
        "- For going-out or travel questions, start from the cleanest stop and name the peak to avoid.",
        "- If they have not said who they are or how they travel, answer what you can and ask one short question.",
        "- Mention a mask only when the band calls for one; keep indoor advice to a sentence.",
        "- If the question is not about air or travel, answer it briefly through the lens of this reading.",
        "- Treat the 80% range as the honest uncertainty; never promise the forecast is exact.",
    ]
    return "\n".join(line for line in lines if line is not None)


def build_messages(message, history=None, context=None, profile="Healthy Adult"):
    messages = [{"role": "system", "content": system_prompt(context, profile)}]
    for turn in (history or [])[-MAX_HISTORY:]:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = str(turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content[:MAX_MESSAGE_CHARS]})
    messages.append({"role": "user", "content": str(message).strip()[:MAX_MESSAGE_CHARS]})
    return messages


def _post(model, messages, key, timeout=TIMEOUT_SECONDS):
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": MAX_REPLY_TOKENS,
    }).encode("utf-8")
    request = urllib.request.Request(OPENROUTER_URL, data=body, method="POST", headers={
        "Authorization": "Bearer " + key,
        "Content-Type": "application/json",
        "HTTP-Referer": "https://delhi-aqi-prediction.vercel.app",
        "X-Title": "Anand Vihar Air Bulletin",
    })
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _reply_text(payload):
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    if isinstance(content, list):
        content = " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content or "").strip()


def answer(message, history=None, context=None, profile="Healthy Adult"):
    """Ask the first model that answers, and say which one did."""
    text = str(message or "").strip()
    if not text:
        raise ChatUnavailable(400, "Ask a question first.")
    key = api_key()
    if not key:
        raise ChatUnavailable(
            503, "The chat is not configured: add OPENROUTER_API_KEY to the server environment.")
    messages = build_messages(text, history, context, profile)
    problems = []
    for model in models():
        try:
            payload = _post(model, messages, key)
        except urllib.error.HTTPError as exc:
            problems.append(f"{model} -> HTTP {exc.code}")
            if exc.code in (401, 403):
                raise ChatUnavailable(503, "OpenRouter rejected the API key.") from exc
            if exc.code == 402:
                raise ChatUnavailable(503, "The OpenRouter account has no credit left for this model.") from exc
            logger.warning("OpenRouter refused %s: HTTP %s", model, exc.code)
            continue
        except Exception as exc:
            problems.append(f"{model} -> {exc}")
            logger.warning("OpenRouter call to %s failed: %s", model, exc)
            continue
        reply = _reply_text(payload)
        if reply:
            return {"reply": reply, "model": payload.get("model") or model}
        problems.append(f"{model} -> empty reply")
    logger.warning("No OpenRouter model answered: %s", "; ".join(problems))
    raise ChatUnavailable(
        502, "The assistant could not answer just now. Check the OpenRouter key and try again.")
