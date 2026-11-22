"""
tracing.py — optional LangFuse tracing for the LangGraph support agent.

One trace per customer query ("support-agent"), with a child span per graph
node (classify, tool, evaluate, respond, escalate), per tool call
(policy_kb, order_tracker, account_lookup) and a generation per LLM call.

Privacy: the trace input/output are set explicitly from the PII-anonymized
query and answer. Spans below the root only see graph state, which already
holds the anonymized query. LLM generations do not record prompt messages
(the input guardrail sends the raw query to the model), only model, output
and token usage.

If the langfuse SDK is missing or keys are not set, everything degrades to
no-ops so the agent, tests and eval harness run unchanged.
"""

import os

_KEYS_SET = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))

try:
    if not _KEYS_SET:
        raise RuntimeError("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set")
    from langfuse.decorators import observe, langfuse_context
    LANGFUSE_ENABLED = True
except Exception:
    LANGFUSE_ENABLED = False

    def observe(func=None, **kwargs):
        if func is not None:
            return func

        def decorator(f):
            return f
        return decorator

    class langfuse_context:  # minimal no-op stand-in
        @staticmethod
        def get_current_trace_id():
            return None

        @staticmethod
        def update_current_trace(**kwargs):
            return None

        @staticmethod
        def update_current_observation(**kwargs):
            return None

        @staticmethod
        def flush():
            return None


def in_trace() -> bool:
    """True when called inside an active LangFuse trace."""
    if not LANGFUSE_ENABLED:
        return False
    try:
        return langfuse_context.get_current_trace_id() is not None
    except Exception:
        return False


def safe_update_trace(**kwargs) -> None:
    if not LANGFUSE_ENABLED:
        return
    try:
        langfuse_context.update_current_trace(**kwargs)
    except Exception:
        pass


def safe_update_observation(**kwargs) -> None:
    if not LANGFUSE_ENABLED:
        return
    try:
        langfuse_context.update_current_observation(**kwargs)
    except Exception:
        pass


def current_trace_id():
    if not LANGFUSE_ENABLED:
        return None
    try:
        return langfuse_context.get_current_trace_id()
    except Exception:
        return None


def flush() -> None:
    if not LANGFUSE_ENABLED:
        return
    try:
        langfuse_context.flush()
    except Exception:
        pass
