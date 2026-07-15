"""
test_llm_fallback.py — Gap 2 unit tests for the Project B fallback helper.

Run from the project root (novus-support-agent/):
    python -m pytest scripts/test_llm_fallback.py -v
    # or, no pytest:
    python scripts/test_llm_fallback.py

Covers the raw-OpenAI path (LITELLM_ENABLED=False) — the branch that previously
had no fallback on the agent/guardrail call sites.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import scripts.llm_fallback as lf


class _FakeMessage:
    def __init__(self, content): self.content = content
class _FakeChoice:
    def __init__(self, content): self.message = _FakeMessage(content)
class _FakeResponse:
    def __init__(self, content): self.choices = [_FakeChoice(content)]


def _make_fake_client(fail_models):
    calls = []
    class _Completions:
        def create(self, *, model, **kwargs):
            calls.append(model)
            if model in fail_models:
                raise RuntimeError(f"simulated outage for {model}")
            return _FakeResponse(f"ok:{model}")
    class _Chat: completions = _Completions()
    class _Client: chat = _Chat()
    return _Client(), calls


def test_primary_succeeds(monkeypatch):
    monkeypatch.setattr(lf, "LITELLM_ENABLED", False)
    client, calls = _make_fake_client(set())
    monkeypatch.setattr(lf, "_client", client)
    assert lf.llm_complete([{"role": "user", "content": "hi"}]) == f"ok:{lf.PRIMARY_MODEL}"
    assert calls == [lf.PRIMARY_MODEL]


def test_falls_back_on_primary_outage(monkeypatch):
    monkeypatch.setattr(lf, "LITELLM_ENABLED", False)
    client, calls = _make_fake_client({lf.PRIMARY_MODEL})
    monkeypatch.setattr(lf, "_client", client)
    assert lf.llm_complete([{"role": "user", "content": "hi"}]) == f"ok:{lf.FALLBACK_MODEL}"
    assert calls == [lf.PRIMARY_MODEL, lf.FALLBACK_MODEL]


def test_raises_when_both_models_down(monkeypatch):
    monkeypatch.setattr(lf, "LITELLM_ENABLED", False)
    client, calls = _make_fake_client({lf.PRIMARY_MODEL, lf.FALLBACK_MODEL})
    monkeypatch.setattr(lf, "_client", client)
    try:
        lf.llm_complete([{"role": "user", "content": "hi"}])
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    assert calls == [lf.PRIMARY_MODEL, lf.FALLBACK_MODEL]


if __name__ == "__main__":
    class _MP:
        def __init__(self): self._u = []
        def setattr(self, o, n, v): self._u.append((o, n, getattr(o, n))); setattr(o, n, v)
        def undo(self):
            for o, n, v in reversed(self._u): setattr(o, n, v)
            self._u = []
    for fn in [test_primary_succeeds, test_falls_back_on_primary_outage, test_raises_when_both_models_down]:
        mp = _MP()
        try:
            fn(mp); print(f"  PASS  {fn.__name__}")
        finally:
            mp.undo()
    print("✅ all fallback tests passed")
