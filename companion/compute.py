"""Explicit per-task compute routing with provenance and capability checks.

Compute is hybrid by design: jetson_local, studio_local, studio_gateway and
cloud are all normal candidates. Routing is EXPLICIT: config names, per task,
a primary backend id and an ordered fallback list. Nothing here picks a
backend on its own judgement.

For each candidate, in order:
  1. no adapter wired for that id            -> skipped ("no adapter")
  2. credential env var named but not set    -> skipped ("not configured");
     only the variable NAME is ever logged, never a value
  3. adapter.capabilities() lacks a need     -> refused BEFORE any call
  4. call raises BackendUnavailable / ConnectionError / TimeoutError / OSError
                                             -> next candidate
The first success returns (result, Provenance). If every candidate fails,
BackendUnavailable is raised carrying the per-candidate attempt list.

Provenance records what the endpoint REPORTS. A Studio endpoint is not assumed
local: Ollama can serve cloud models through a signed-in local server, so an
adapter that cannot tell sets where_computed="unknown".

The command guard never goes through this module: physical control and
stopping do not depend on any inference backend.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Tuple

from .adapters.base import BackendUnavailable

TASKS = ("conversation", "vision_interpret", "speech_to_text", "text_to_speech",
         "note_retrieval")
KINDS = frozenset({"jetson_local", "studio_local", "studio_gateway", "cloud"})
CAPABILITIES = frozenset({"text", "image", "audio", "tools", "streaming"})
WHERE = frozenset({"jetson", "studio", "cloud", "unknown"})

# What each task needs from a backend.
TASK_NEEDS: Dict[str, FrozenSet[str]] = {
    "conversation": frozenset({"text"}),
    "vision_interpret": frozenset({"image"}),
    "speech_to_text": frozenset({"audio"}),
    "text_to_speech": frozenset({"audio"}),
    "note_retrieval": frozenset({"text"}),
}


class ConfigError(ValueError):
    pass


@dataclass
class BackendSpec:
    id: str
    kind: str
    provider: str = ""
    model: str = ""
    endpoint: str = ""
    credential_env: str = ""  # NAME of an env var; never a value

    @classmethod
    def from_dict(cls, bid: str, d: Mapping[str, Any]) -> "BackendSpec":
        allowed = {"kind", "provider", "model", "endpoint", "credential_env", "notes"}
        extra = set(d) - allowed
        if extra:
            raise ConfigError("backend %s: unknown keys %s" % (bid, sorted(extra)))
        kind = d.get("kind")
        if kind not in KINDS:
            raise ConfigError("backend %s: kind must be one of %s" % (bid, sorted(KINDS)))
        cred = str(d.get("credential_env") or "")
        if cred and not cred.replace("_", "").isalnum():
            raise ConfigError("backend %s: credential_env must be a variable NAME" % bid)
        return cls(id=bid, kind=kind, provider=str(d.get("provider") or ""),
                   model=str(d.get("model") or ""), endpoint=str(d.get("endpoint") or ""),
                   credential_env=cred)


@dataclass
class Provenance:
    backend_id: str
    kind: str
    provider: str
    model: str
    where_computed: str
    latency_ms: float
    cold_start: Optional[bool]  # None = not observable
    task: str = ""
    attempts: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComputeConfig:
    backends: Dict[str, BackendSpec]
    routes: Dict[str, List[str]]  # task -> [primary, fallback1, ...]

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ComputeConfig":
        if not isinstance(d, Mapping):
            raise ConfigError("config must be a mapping")
        raw_b = d.get("backends") or {}
        backends = {bid: BackendSpec.from_dict(bid, spec) for bid, spec in raw_b.items()}
        routes: Dict[str, List[str]] = {}
        for task, slot in (d.get("tasks") or {}).items():
            if task not in TASKS:
                raise ConfigError("unknown task %r" % task)
            primary = slot.get("backend")
            order = ([primary] if primary else []) + list(slot.get("fallbacks") or [])
            for bid in order:
                if bid not in backends:
                    raise ConfigError("task %s names undefined backend %r" % (task, bid))
            routes[task] = order
        return cls(backends=backends, routes=routes)


def load_compute_config(path: str) -> ComputeConfig:
    """Load config/compute.yaml. pyyaml is imported lazily (present on the Jetson)."""
    if path.endswith(".json"):
        import json
        with open(path, "r", encoding="utf-8") as f:
            return ComputeConfig.from_dict(json.load(f))
    import yaml  # lazy: core package stays stdlib-only at import time
    with open(path, "r", encoding="utf-8") as f:
        return ComputeConfig.from_dict(yaml.safe_load(f))


class Router:
    def __init__(self, config: ComputeConfig, adapters: Mapping[str, Any],
                 env: Optional[Mapping[str, str]] = None,
                 clock: Callable[[], float] = time.monotonic,
                 log: Optional[Callable[[str, Dict[str, Any]], None]] = None) -> None:
        self.config = config
        self.adapters = dict(adapters)
        self.env = os.environ if env is None else env
        self.clock = clock
        self.log = log

    def status(self, backend_id: str) -> str:
        """'ready', 'no adapter', 'not configured (VAR unset)' or 'undefined'."""
        spec = self.config.backends.get(backend_id)
        if spec is None:
            return "undefined"
        if spec.credential_env and not self.env.get(spec.credential_env):
            return "not configured (%s unset)" % spec.credential_env
        if backend_id not in self.adapters:
            return "no adapter"
        return "ready"

    def call(self, task: str, method: str, *args: Any,
             needs: Optional[FrozenSet[str]] = None) -> Tuple[Any, Provenance]:
        if task not in TASKS:
            raise ValueError("unknown task %r" % task)
        need = TASK_NEEDS[task] if needs is None else frozenset(needs)
        attempts: List[Dict[str, str]] = []
        for bid in self.config.routes.get(task, []):
            spec = self.config.backends[bid]
            st = self.status(bid)
            if st != "ready":
                attempts.append({"backend_id": bid, "outcome": st})
                continue
            adapter = self.adapters[bid]
            caps = frozenset(adapter.capabilities())
            missing = need - caps
            if missing:
                attempts.append({"backend_id": bid,
                                 "outcome": "capability mismatch: lacks %s" % ",".join(sorted(missing))})
                continue
            t0 = self.clock()
            try:
                result = getattr(adapter, method)(*args)
            except (BackendUnavailable, ConnectionError, TimeoutError, OSError) as exc:
                attempts.append({"backend_id": bid, "outcome": "failed: %s" % type(exc).__name__})
                continue
            latency = (self.clock() - t0) * 1000.0
            rep = dict(adapter.report()) if hasattr(adapter, "report") else {}
            where = rep.get("where_computed", "unknown")
            prov = Provenance(backend_id=bid, kind=spec.kind,
                              provider=str(rep.get("provider") or spec.provider),
                              model=str(rep.get("model") or spec.model),
                              where_computed=where if where in WHERE else "unknown",
                              latency_ms=round(latency, 3),
                              cold_start=rep.get("cold_start"), task=task,
                              attempts=attempts + [{"backend_id": bid, "outcome": "ok"}])
            if self.log:
                self.log("compute:" + task, prov.to_dict())
            return result, prov
        if self.log:
            self.log("compute:unavailable", {"task": task, "attempts": attempts})
        err = BackendUnavailable("%s unavailable: %s" % (task, attempts or "no route"),
                                 what=task)
        err.attempts = attempts  # type: ignore[attr-defined]
        raise err


class RoutedBrain:
    """Brain interface backed by the Router. Keeps the last Provenance for logging."""

    def __init__(self, router: Router) -> None:
        self.router = router
        self.last_provenance: Optional[Provenance] = None

    def respond(self, text: str, context: List[str]):
        result, self.last_provenance = self.router.call("conversation", "respond", text, context)
        return result

    def describe_image(self, path: str, prompt: str) -> str:
        result, self.last_provenance = self.router.call("vision_interpret", "describe_image",
                                                        path, prompt)
        return result
