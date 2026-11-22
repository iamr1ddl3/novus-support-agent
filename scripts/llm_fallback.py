"""
llm_fallback.py — Gap 2: one model-call path with fallback, shared by every
non-instructor LLM call site in the Novus Support Agent (agent nodes, guardrails,
classifier, eval judges).

Mirrors the Project A helper. Depends only on the leaf scripts.tracing module, so any
module (including agent.py and the guardrails it loads) can import it without a
cycle.

Resilience policy:
  - LiteLLM installed  → litellm.completion(..., fallbacks=[FALLBACK_MODEL]).
  - LiteLLM absent      → raw OpenAI; on ANY exception, retry once on FALLBACK_MODEL.

A primary-model outage or rate-limit degrades to the fallback model instead of
taking down the agent path / guardrails / classification (the original gap:
"agent.py + guardrails have zero fallback").
"""

import os
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

from scripts.tracing import observe, in_trace, safe_update_observation

load_dotenv()

PRIMARY_MODEL = "gpt-4o-mini"
FALLBACK_MODEL = "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0.0

try:
    import litellm
    litellm.set_verbose = False
    LITELLM_ENABLED = True
except ImportError:
    LITELLM_ENABLED = False

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _complete(
    messages: list[dict],
    model: str | None = None,
    *,
    response_format: dict | None = None,
    max_tokens: int | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
) -> tuple[str, str, dict | None]:
    """Run the model call; return (content, model actually used, usage dict)."""
    selected_model = model or PRIMARY_MODEL
    kwargs: dict[str, Any] = {"messages": messages, "temperature": temperature}
    if response_format is not None:
        kwargs["response_format"] = response_format
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    if LITELLM_ENABLED:
        response = litellm.completion(
            model=selected_model,
            fallbacks=[FALLBACK_MODEL],
            **kwargs,
        )
        return _unpack(response, selected_model)

    try:
        response = _client.chat.completions.create(model=selected_model, **kwargs)
    except Exception:
        response = _client.chat.completions.create(model=FALLBACK_MODEL, **kwargs)
        return _unpack(response, FALLBACK_MODEL)
    return _unpack(response, selected_model)


def _unpack(response: Any, requested_model: str) -> tuple[str, str, dict | None]:
    content = response.choices[0].message.content.strip()
    model_used = getattr(response, "model", None) or requested_model
    usage = None
    u = getattr(response, "usage", None)
    if u is not None:
        usage = {
            "input": getattr(u, "prompt_tokens", None),
            "output": getattr(u, "completion_tokens", None),
            "total": getattr(u, "total_tokens", None),
        }
    return content, model_used, usage


# LangFuse generation: records model, output and token usage, but not the
# prompt messages (the input guardrail sends the raw, un-anonymized query).
@observe(name="llm", as_type="generation", capture_input=False, capture_output=False)
def _traced_complete(messages: list[dict], model: str | None, **kwargs: Any) -> str:
    content, model_used, usage = _complete(messages, model, **kwargs)
    safe_update_observation(model=model_used, output=content, usage=usage)
    return content


def llm_complete(
    messages: list[dict],
    model: str | None = None,
    *,
    response_format: dict | None = None,
    max_tokens: int | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """Return the completion content string, with single-fallback resilience.

    Traced as a LangFuse generation only when called inside an agent trace, so
    eval judges and other standalone callers never create stray traces.
    """
    kwargs = {"response_format": response_format, "max_tokens": max_tokens,
              "temperature": temperature}
    if in_trace():
        return _traced_complete(messages, model, **kwargs)
    return _complete(messages, model, **kwargs)[0]
