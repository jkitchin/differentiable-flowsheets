"""Forward one assistant question to the Anthropic API.

The third of the panel's three providers, and the only one that needs
the server at all. WebLLM answers inside the page and an
OpenAI-compatible endpoint is reached from the page directly; neither
involves this file. This one exists so that a key can be held by the
process the user started, in its own environment, instead of being
typed into a web page and kept in browser storage.

Two things are deliberate. The reply is not streamed --- the answer to
a brief is a paragraph or two and the editor's server is a
``http.server``, so a token stream would buy little and cost the only
simple thing about this module. And nothing here is enabled by
default: with no ``ANTHROPIC_API_KEY`` in the environment the route
refuses and says so, which is the correct behaviour for the one
provider that sends the brief off the machine.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

API = "https://api.anthropic.com/v1/messages"
VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024
TIMEOUT = 90.0

#: Refuse a brief larger than this rather than spend on it. The packs
#: `context.py` builds are budgeted at a few thousand tokens, so
#: anything near this is a bug on the way to becoming a bill.
MAX_CHARS = 200_000


def configured() -> bool:
    """Whether a key is present. The panel asks before offering the option."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _text(payload: dict) -> str:
    """The assistant's words out of a Messages response."""
    return "".join(
        block.get("text", "")
        for block in payload.get("content", [])
        if block.get("type") == "text"
    )


def answer(messages: list[dict], *, model: str | None = None,
           max_tokens: int = MAX_TOKENS) -> dict:
    """Answer one question, or say why not.

    Args:
        messages: chat turns as the page sends them, in OpenAI shape ---
            a leading ``system`` turn plus one ``user`` turn. The system
            turn is lifted out into the API's own ``system`` field.
        model: model id. Defaults to ``DIFFLOW_ASSISTANT_MODEL`` in the
            environment, then :data:`DEFAULT_MODEL`.
        max_tokens: cap on the reply.

    Returns:
        ``{"ok": True, "text": ..., "model": ...}``, or
        ``{"ok": False, "error": ...}``. Every failure is an answer with
        a 200: a missing key, a rejected request and a timeout are all
        things the panel should show the user, not stack traces.
    """
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return {"ok": False, "error":
                "no ANTHROPIC_API_KEY in the environment of the process serving "
                "this editor. Set it and restart, or choose a provider that "
                "answers locally."}

    system = "\n\n".join(m.get("content", "") for m in messages
                         if m.get("role") == "system")
    turns = [{"role": m.get("role"), "content": m.get("content", "")}
             for m in messages if m.get("role") in ("user", "assistant")]
    if not turns:
        return {"ok": False, "error": "nothing to ask: no user turn"}
    size = len(system) + sum(len(t["content"]) for t in turns)
    if size > MAX_CHARS:
        return {"ok": False, "error":
                f"the brief is {size} characters, over the {MAX_CHARS} this "
                "route will forward"}

    model = model or os.environ.get("DIFFLOW_ASSISTANT_MODEL") or DEFAULT_MODEL
    body = {"model": model, "max_tokens": max_tokens, "messages": turns}
    if system:
        body["system"] = system
    request = urllib.request.Request(
        API,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json",
                 "x-api-key": key,
                 "anthropic-version": VERSION},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        try:
            detail = json.loads(detail)["error"]["message"]
        except Exception:
            pass
        return {"ok": False, "error": f"the API refused ({exc.code}): {detail}"}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    text = _text(payload)
    if not text:
        return {"ok": False, "error":
                f"the API answered with no text (stop reason "
                f"{payload.get('stop_reason')!r})"}
    return {"ok": True, "text": text, "model": payload.get("model", model),
            "usage": payload.get("usage")}
