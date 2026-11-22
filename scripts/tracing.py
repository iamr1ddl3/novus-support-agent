"""
tracing.py — optional LangFuse tracing for the LangGraph support agent.

One trace per customer query ("support-agent"), with a child span per graph
node (classify, tool, evaluate, respond, escalate), per tool call
(policy_kb, order_tracker, account_lookup) and a generation per LLM call.

Uses the LangFuse Python SDK v4+/v5, which exports traces over OpenTelemetry.
(LangFuse Cloud shut down the legacy v2 ingestion API on 2026-11-16, so SDK
2.x traces are silently dropped.)

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
_client = None

try:
    if not _KEYS_SET:
        raise RuntimeError("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set")
    from langfuse import observe, get_client, propagate_attributes
    _client = get_client()
    LANGFUSE_ENABLED = True
except Exception:
    LANGFUSE_ENABLED = False

    def observe(func=None, **kwargs):
        if func is not None:
            return func

        def decorator(f):
            return f
        return decorator


def in_trace() -> bool:
    """True when called inside an active LangFuse trace."""
    if not LANGFUSE_ENABLED:
        return False
    try:
        from opentelemetry import trace as _otel_trace
        return _otel_trace.get_current_span().get_span_context().is_valid
    except Exception:
        return False


def _meta_str(value) -> str:
    # propagate_attributes only accepts short string values.
    s = value if isinstance(value, str) else str(value)
    return s[:200]


def safe_update_trace(input=None, output=None, tags=None, metadata=None) -> None:
    """Set trace-level input/output (on the root span), tags and metadata.

    Call from inside the root @observe function.
    """
    if not LANGFUSE_ENABLED:
        return
    try:
        _client.update_current_span(input=input, output=output, metadata=metadata)
        attrs = {}
        if tags:
            attrs["tags"] = [str(t) for t in tags]
        if metadata:
            attrs["metadata"] = {str(k): _meta_str(v) for k, v in metadata.items()}
        if attrs:
            with propagate_attributes(**attrs):
                pass
    except Exception:
        pass


def safe_update_observation(model=None, output=None, usage=None, **kwargs) -> None:
    """Update the current generation (model, output, token usage)."""
    if not LANGFUSE_ENABLED:
        return
    try:
        usage_details = None
        if usage:
            usage_details = {k: int(v) for k, v in usage.items() if v is not None}
        _client.update_current_generation(
            model=model, output=output, usage_details=usage_details, **kwargs
        )
    except Exception:
        pass


def current_trace_id():
    if not LANGFUSE_ENABLED:
        return None
    try:
        return _client.get_current_trace_id()
    except Exception:
        return None


def flush() -> None:
    if not LANGFUSE_ENABLED:
        return
    try:
        _client.flush()
    except Exception:
        pass
