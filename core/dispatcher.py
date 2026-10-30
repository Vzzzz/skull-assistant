"""Mode dispatcher (D-030, T-044): a thin layer over systemd user units, llama-server slots and Redis.

The manifests (core.manifests, validated at start) say which service is resident in which mode and how it freezes /
thaws; this process only applies them:
  - state in Redis: mode:current (A by default), mode:transition (while a switch runs, TTL), mode:slot:<mode>
    (the slot file saved when leaving that mode and the dialog:history hash it was saved at);
  - commands: stream cmd:mode {mode, reason, id}; events: stream events:mode {id, from, to, phase, service, t_ms, dt_ms}
    for every phase (brain, TTS masking U-003 and the bench read them);
  - switch X -> Y: (a) freeze every service resident in X and evacuated in Y, plus every service whose model differs
    between X and Y (manifest `modes`, e.g. brain), GPU holders (`cuda_context`) first: `slot-save-kill` = POST
    /slots/0?action=save (file <mode>-<history hash>) + systemctl stop, `kill` = stop; (b) mode:current = Y;
    (c) thaw every service resident in Y that is not running: start, wait for the healthcheck (URL, else the unit
    active) up to thaw_budget_s (null -> 30 s); `start-slot-restore` restores the slot saved when Y was left, only
    if dialog:history is unchanged since; (d) never_evict is never touched; (e) any error -> the same switch back to X,
    events:mode {phase: failed}.
  - units installed but not enabled (systemctl is-enabled != enabled) are not production (skull-vision after T-031):
    never started or stopped here.
On start: reconcile — the residents of mode:current running (and skull-llm holding that mode's model), the rest
stopped.

  .venv/bin/python -m core.dispatcher                 # the service (unit skull-dispatcher)
  .venv/bin/python -m core.dispatcher --switch B      # ask the running dispatcher, print the phases
  .venv/bin/python -m core.dispatcher --status
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import redis
import requests

from core.manifests import ROOT, validate

r = redis.Redis()
CURRENT, TRANSITION, CMD, EVENTS = "mode:current", "mode:transition", "cmd:mode", "events:mode"
HISTORY = "dialog:history"
TRANSITION_TTL_S = 300
DEFAULT_THAW_S = 30


def systemctl(*a):
    return subprocess.run(["systemctl", "--user", *a], capture_output=True, text=True)


def is_active(unit):
    return systemctl("is-active", unit).stdout.strip() == "active"


def is_enabled(unit):
    return systemctl("is-enabled", unit).stdout.strip() == "enabled"


def history_hash():
    h = hashlib.sha1()
    for x in r.lrange(HISTORY, 0, -1):
        h.update(x)
    return h.hexdigest()[:12]


class Switch:
    """One transition: phases go to events:mode with times from its start."""

    def __init__(self, src, dst, reason, req_id):
        self.src, self.dst, self.reason, self.id = src, dst, reason, req_id
        self.t0 = time.time()

    def event(self, phase, service="", dt=None, **extra):
        f = {"id": self.id, "from": self.src, "to": self.dst, "phase": phase, "service": service,
             "t_ms": f"{1000 * (time.time() - self.t0):.0f}", "ts": f"{time.time():.3f}"}
        if dt is not None:
            f["dt_ms"] = f"{1000 * dt:.0f}"
        f.update({k: str(v) for k, v in extra.items()})
        r.xadd(EVENTS, f, maxlen=10000, approximate=True)
        print(f"dispatcher: [{self.id}] {self.src}->{self.dst} {phase} {service} "
              + (f"{1000 * dt:.0f} ms " if dt is not None else "") + f"(t {f['t_ms']} ms)"
              + "".join(f" {k}={v}" for k, v in extra.items()), flush=True)


class Dispatcher:
    def __init__(self, target=None):
        self.manifests, self.modes, self.target = validate(target)

    # --- what a service is in a mode
    def resident(self, m, mode):
        return m.get("never_evict") or m["residency"][mode] == "resident"

    def model_changes(self, m, src, dst):
        """Resident in both modes but another model (manifest `modes`): it has to be cycled."""
        md = m.get("modes") or {}
        strip = lambda x: {k: v for k, v in (x or {}).items() if k != "budget"}
        return self.resident(m, src) and self.resident(m, dst) and strip(md.get(src)) != strip(md.get(dst))

    def units(self, m):
        """Units this service controls in the order they stop (the model holder first)."""
        return [u for u in (m.get("server_unit"), m["unit"]) if u]

    def managed(self, m):
        return not m.get("never_evict") and all(is_enabled(u) for u in self.units(m))

    def base_url(self, m):
        u = urlparse(m["healthcheck"])
        return f"{u.scheme}://{u.netloc}"

    # --- lifecycle verbs
    def freeze(self, sw, name, m, cycle_only):
        """Stop the service as its manifest says; cycle_only: only the model holder (the responder keeps running)."""
        lc = m["lifecycle"]
        if lc["freeze"] == "slot-save-kill" and m.get("healthcheck"):
            t = time.time()
            fname, h = f"{sw.src}-{history_hash()}.bin", history_hash()
            try:
                resp = requests.post(f"{self.base_url(m)}/slots/0?action=save", json={"filename": fname}, timeout=30)
                resp.raise_for_status()
                r.set(f"mode:slot:{sw.src}", json.dumps({"file": fname, "hash": h}))
                sw.event("save", name, time.time() - t, file=fname)
            except requests.RequestException as e:      # nothing to save is not a reason to stay
                sw.event("save_skipped", name, time.time() - t, error=type(e).__name__)
        elif lc["freeze"] not in ("kill", "slot-save-kill"):
            raise NotImplementedError(f"freeze {lc['freeze']} ({name})")
        units = [m["server_unit"]] if cycle_only and m.get("server_unit") else self.units(m)
        for u in units:
            t = time.time()
            res = systemctl("stop", u)
            if res.returncode:
                raise RuntimeError(f"systemctl stop {u}: {res.stderr.strip()}")
            sw.event("stop", name, time.time() - t, unit=u)

    def thaw(self, sw, name, m, mode):
        lc = m["lifecycle"]
        if lc["thaw"] not in ("start", "start-slot-restore"):
            raise NotImplementedError(f"thaw {lc['thaw']} ({name})")
        t = time.time()
        for u in reversed(self.units(m)):                  # the model holder first, then its client
            if not is_active(u):
                res = systemctl("start", u)
                if res.returncode:
                    raise RuntimeError(f"systemctl start {u}: {res.stderr.strip()}")
        sw.event("start", name, time.time() - t)
        budget = ((m.get("modes") or {}).get(mode) or {}).get("thaw_budget_s") or lc.get("thaw_budget_s") or DEFAULT_THAW_S
        t_h = time.time()
        while True:
            dead = [u for u in self.units(m) if systemctl("is-active", u).stdout.strip() in ("failed", "activating", "inactive")]
            if dead:                                       # exited while loading (Restart=on-failure -> activating)
                raise RuntimeError(f"{name}: {', '.join(dead)} not running after start")
            if m.get("healthcheck"):
                try:
                    if requests.get(m["healthcheck"], timeout=2).status_code == 200:
                        break
                except requests.RequestException:
                    pass
            elif all(is_active(u) for u in self.units(m)):
                break
            if time.time() - t_h > max(budget, DEFAULT_THAW_S) * 3:     # the budget is a measurement, not a timeout
                raise TimeoutError(f"{name}: not healthy after {time.time() - t_h:.0f} s")
            time.sleep(0.1)
        dt = time.time() - t
        sw.event("health", name, dt, budget_s=budget, over=int(dt > budget))
        if lc["thaw"] == "start-slot-restore":
            saved = json.loads(r.get(f"mode:slot:{mode}") or "{}")
            h = history_hash()
            if saved and saved.get("hash") == h:
                t = time.time()
                try:
                    requests.post(f"{self.base_url(m)}/slots/0?action=restore", json={"filename": saved["file"]},
                                  timeout=60).raise_for_status()
                    sw.event("restore", name, time.time() - t, file=saved["file"])
                except requests.RequestException as e:
                    sw.event("restore_failed", name, time.time() - t, error=type(e).__name__)
            else:
                sw.event("restore_skipped", name, reason="history changed" if saved else "no slot")

    # --- transitions
    def plan(self, src, dst):
        freeze, thaw = [], []
        for n, m in self.manifests.items():
            if not self.managed(m):
                continue
            if self.resident(m, src) and not self.resident(m, dst):
                freeze.append((n, m, False))
            elif self.model_changes(m, src, dst):
                freeze.append((n, m, True))
            if self.resident(m, dst):
                thaw.append((n, m))
        freeze.sort(key=lambda x: not x[1].get("cuda_context"))     # free the GPU before anything starts
        thaw.sort(key=lambda x: not x[1].get("cuda_context"))
        return freeze, thaw

    def switch(self, dst, reason="", req_id=None, rollback=False):
        src = (r.get(CURRENT) or b"A").decode()
        sw = Switch(src, dst, reason, req_id or uuid.uuid4().hex[:8])
        if dst == src:
            sw.event("noop")
            return True
        r.set(TRANSITION, json.dumps({"from": src, "to": dst, "id": sw.id, "t": sw.t0}), ex=TRANSITION_TTL_S)
        sw.event("begin", reason=reason or "-", rollback=int(rollback))
        try:
            freeze, thaw = self.plan(src, dst)
            for n, m, cycle_only in freeze:
                self.freeze(sw, n, m, cycle_only)
            r.set(CURRENT, dst)
            sw.event("switch")
            for n, m in thaw:
                if m.get("server_unit") and not is_active(m["server_unit"]) or not all(is_active(u) for u in self.units(m)):
                    self.thaw(sw, n, m, dst)
            sw.event("done")
            return True
        except Exception as e:                                 # noqa: BLE001 — any failure rolls back
            sw.event("failed", error=f"{type(e).__name__}: {e}"[:300])
            if rollback:
                return False
            r.set(CURRENT, dst)                                 # the rollback starts from where we got to
            self.switch(src, reason=f"rollback of {sw.id}", req_id=sw.id + "-rb", rollback=True)
            return False
        finally:
            r.delete(TRANSITION)

    def reconcile(self):
        """mode:current (A if none): its residents running, skull-llm holding its model, the rest stopped."""
        mode = (r.get(CURRENT) or b"A").decode()
        r.set(CURRENT, mode)
        r.delete(TRANSITION)
        sw = Switch(mode, mode, "reconcile", "rc-" + uuid.uuid4().hex[:6])
        sw.event("reconcile_begin")
        for n, m in self.manifests.items():
            if not self.managed(m):
                if not m.get("never_evict"):
                    sw.event("not_managed", n, reason="unit not enabled")
                continue
            if self.resident(m, mode):
                stale = m.get("server_unit") and (r.get("mode:llm") or b"A").decode() != mode and is_active(m["server_unit"])
                if stale:                                        # the server holds another mode's model
                    self.freeze(sw, n, m, True)
                if stale or not all(is_active(u) for u in self.units(m)):
                    self.thaw(sw, n, m, mode)
            elif any(is_active(u) for u in self.units(m)):
                self.freeze(sw, n, m, False)
        sw.event("reconcile_done")

    def run(self):
        print(f"dispatcher: {len(self.manifests)} manifests ok on {self.target}", flush=True)
        self.reconcile()
        last = "$"
        while True:
            ev = r.xread({CMD: last}, count=1, block=5000)
            if not ev:
                continue
            last, f = ev[0][1][0]
            dst = f.get(b"mode", b"").decode()
            if dst not in ("A", "B"):
                print(f"dispatcher: ignored command {f}", flush=True)
                continue
            self.switch(dst, f.get(b"reason", b"").decode(), f.get(b"id", b"").decode() or None)


def ask(dst, reason, timeout_s=300):
    """Send cmd:mode to the running dispatcher and print its events until done / failed (and the rollback)."""
    req = uuid.uuid4().hex[:8]
    last = r.xinfo_stream(EVENTS)["last-generated-id"] if r.exists(EVENTS) else "0-0"
    r.xadd(CMD, {"mode": dst, "reason": reason, "id": req})
    t_end, failed = time.time() + timeout_s, False
    while time.time() < t_end:
        ev = r.xread({EVENTS: last}, count=50, block=1000)
        for _, items in ev or []:
            for last, f in items:
                f = {k.decode(): v.decode() for k, v in f.items()}
                if not f.get("id", "").startswith(req):
                    continue
                print(f"{f['t_ms']:>7} ms  {f['phase']:<16} {f.get('service', ''):<10} "
                      + (f"{f['dt_ms']} ms" if "dt_ms" in f else "")
                      + "".join(f" {k}={v}" for k, v in f.items() if k in ("file", "unit", "reason", "error", "over")))
                if f["phase"] in ("done", "noop") or (f["phase"] == "failed" and f["id"].endswith("-rb")):
                    return f["phase"] in ("done", "noop") and not failed     # a rollback that worked is still a failure
                if f["phase"] == "failed" and not f["id"].endswith("-rb"):
                    failed = True
                    print("  rolling back...")
    print("timeout: is skull-dispatcher running?")
    return False


def status():
    d = Dispatcher()
    mode = (r.get(CURRENT) or b"A").decode()
    print(f"mode:current {mode}; transition {r.get(TRANSITION)}; skull-llm holds {r.get('mode:llm')}")
    for n, m in d.manifests.items():
        st = " ".join(f"{u}={systemctl('is-active', u).stdout.strip()}" for u in d.units(m))
        print(f"  {n:<14} {m['residency'][mode]:<10} {'never_evict ' if m.get('never_evict') else ''}"
              f"{'' if d.managed(m) or m.get('never_evict') else 'not managed (unit not enabled) '}{st}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--switch", choices=["A", "B"])
    ap.add_argument("--reason", default="manual")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.switch:
        sys.exit(0 if ask(a.switch, a.reason) else 1)
    if a.status:
        return status()
    Dispatcher().run()


if __name__ == "__main__":
    main()
