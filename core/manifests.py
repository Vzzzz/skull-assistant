"""Service manifests as the single source of truth for modes (D-030, ROADMAP §2 contract, T-044 step 1).

Loads services/*/manifest.yaml and deploy/modes.yaml and validates the §2 rules:
  - required fields; residency per mode is resident | evacuated; lifecycle verbs from the contract;
  - budget.measured_on in every manifest (no admission without it);
  - never_evict on exactly one service, and that service resident in every mode (the dispatcher never touches it);
  - per mode and target: sum of the residents' peaks + the mode's KV reserve <= the mode budget
    (stand: VRAM and RAM separately; Orin: VRAM + RAM against one unified pool).
A service may override its budget per mode (`modes.<M>.budget`, e.g. brain runs another model in B). A violation
raises ManifestError with every problem listed; the dispatcher does not start.

  .venv/bin/python -m core.manifests [--target stand|orin]   # table: mode -> residents -> sums -> budget -> ok / over
"""
import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODES_FILE = ROOT / "deploy/modes.yaml"
MODES = ("A", "B", "C")
FREEZE = ("slot-save-kill", "serialize-kill", "kill")
THAW = ("start-slot-restore", "start-restore-queue", "start")


class ManifestError(Exception):
    pass


def load_manifests(root=ROOT):
    """{name: manifest} from services/*/manifest.yaml."""
    out = {}
    for f in sorted((root / "services").glob("*/manifest.yaml")):
        m = yaml.safe_load(f.read_text(encoding="utf-8"))
        m["_file"] = str(f.relative_to(root))
        out[m.get("name", f.parent.name)] = m
    return out


def load_modes(path=MODES_FILE):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def budget(m, mode):
    """The service's peaks in this mode: modes.<mode>.budget overrides the top-level budget."""
    b = dict(m.get("budget") or {})
    b.update(((m.get("modes") or {}).get(mode) or {}).get("budget") or {})
    return b


def residents(manifests, mode):
    return [n for n, m in manifests.items() if (m.get("residency") or {}).get(mode) == "resident"]


def mode_sums(manifests, mode):
    names = residents(manifests, mode)
    return names, sum(budget(manifests[n], mode).get("vram_mb") or 0 for n in names), \
        sum(budget(manifests[n], mode).get("ram_mb") or 0 for n in names)


def check_mode(manifests, modes, mode, target):
    """-> (names, vram, ram, limit text, ok) for one mode on one target."""
    t = modes["targets"][target]
    kv = (modes["modes"].get(mode) or {}).get("kv_reserve_mb", 0)
    names, vram, ram = mode_sums(manifests, mode)
    vram += kv
    if t.get("unified"):
        return names, vram, ram, f"unified {t['unified_mb']}", vram + ram <= t["unified_mb"]
    return names, vram, ram, f"VRAM {t['vram_mb']}, RAM {t['ram_mb']}", vram <= t["vram_mb"] and ram <= t["ram_mb"]


def problems(manifests, modes, target):
    errs = []
    for n, m in manifests.items():
        where = f"{m['_file']}:"
        for k in ("name", "unit", "cuda_context", "budget", "residency", "lifecycle", "never_evict"):
            if k not in m:
                errs.append(f"{where} no `{k}`")
        b = m.get("budget") or {}
        if not b.get("measured_on"):
            errs.append(f"{where} budget.measured_on is required (§2: no admission without it)")
        for k in ("vram_mb", "ram_mb"):
            if not isinstance(b.get(k), (int, float)):
                errs.append(f"{where} budget.{k} must be a number")
        res = m.get("residency") or {}
        for mode in MODES:
            if res.get(mode) not in ("resident", "evacuated"):
                errs.append(f"{where} residency.{mode} must be resident | evacuated, not {res.get(mode)!r}")
        lc = m.get("lifecycle") or {}
        if lc.get("freeze") not in FREEZE:
            errs.append(f"{where} lifecycle.freeze {lc.get('freeze')!r} not in {FREEZE}")
        if lc.get("thaw") not in THAW:
            errs.append(f"{where} lifecycle.thaw {lc.get('thaw')!r} not in {THAW}")
        if "slot" in f"{lc.get('freeze')}{lc.get('thaw')}" and not m.get("healthcheck"):
            errs.append(f"{where} slot lifecycle needs a healthcheck (the server's URL)")
        for mode in m.get("modes") or {}:
            if mode not in MODES:
                errs.append(f"{where} modes.{mode}: unknown mode")
    keepers = [n for n, m in manifests.items() if m.get("never_evict") is True]
    if len(keepers) != 1:
        errs.append(f"never_evict must be true on exactly one service (the wake word), found {keepers or 'none'}")
    for n in keepers:
        off = [mode for mode in MODES if (manifests[n].get("residency") or {}).get(mode) != "resident"]
        if off:
            errs.append(f"{manifests[n]['_file']}: never_evict service must be resident in every mode, not in {off}")
    if target not in modes.get("targets", {}):
        errs.append(f"deploy/modes.yaml: no target {target!r}")
        return errs
    for mode in MODES:
        if mode not in modes.get("modes", {}):
            errs.append(f"deploy/modes.yaml: no mode {mode}")
            continue
        names, vram, ram, limit, ok = check_mode(manifests, modes, mode, target)
        if not ok:
            errs.append(f"mode {mode} on {target}: residents {names} VRAM {vram} + RAM {ram} MB over the budget ({limit})")
    return errs


def validate(target=None, root=ROOT):
    """-> (manifests, modes, target) or ManifestError listing every violation."""
    modes = load_modes(root / "deploy/modes.yaml")
    target = target or modes.get("target", "stand")
    manifests = load_manifests(root)
    errs = problems(manifests, modes, target)
    if errs:
        raise ManifestError("manifests violate the contract (ROADMAP §2):\n  " + "\n  ".join(errs))
    return manifests, modes, target


def table(manifests, modes, target):
    print(f"target: {target}")
    print(f"{'mode':<4} | {'residents':<60} | {'VRAM':>5} | {'RAM':>5} | budget                  | ok")
    for mode in MODES:
        names, vram, ram, limit, ok = check_mode(manifests, modes, mode, target)
        print(f"{mode:<4} | {', '.join(names):<60} | {vram:>5} | {ram:>5} | {limit:<23} | {'ok' if ok else 'OVER'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", help="stand | orin (default: deploy/modes.yaml target)")
    a = ap.parse_args()
    modes = load_modes()
    target = a.target or modes.get("target", "stand")
    manifests = load_manifests()
    table(manifests, modes, target)
    errs = problems(manifests, modes, target)
    if errs:
        print("\nviolations:\n  " + "\n  ".join(errs))
        sys.exit(1)
    print("\nmanifests ok")


if __name__ == "__main__":
    main()
