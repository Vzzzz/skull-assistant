"""core/manifests.py (T-044): the ROADMAP §2 contract — never_evict on exactly one service and resident everywhere,
measured_on required, the residents' peaks + KV reserve within the mode budget (stand: VRAM and RAM; Orin: one pool),
per-mode budget overrides. Fixture tree: tests/unit/fixtures/manifests (A: 4000+500 KV VRAM, 1800 RAM; B: 7000, 1700)."""
import shutil
from pathlib import Path

import pytest
import yaml

from core.manifests import ManifestError, budget, check_mode, load_manifests, load_modes, residents, validate

FIX = Path(__file__).resolve().parent / "fixtures/manifests"
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "repo"
    shutil.copytree(FIX, root)
    return root


def edit(root, rel, fn):
    p = root / rel
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    fn(d)
    p.write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")


def errors(root, target=None):
    with pytest.raises(ManifestError) as e:
        validate(target, root=root)
    return str(e.value)


def test_fixture_is_valid():
    manifests, modes, target = validate(root=FIX)
    assert target == "stand" and set(manifests) == {"kws", "llm", "voice"}


def test_repo_manifests_are_valid():
    """The real services/*/manifest.yaml against deploy/modes.yaml — what the dispatcher checks at start."""
    manifests, _, _ = validate(root=REPO)
    assert [n for n, m in manifests.items() if m.get("never_evict")] == ["wakeword"]


def test_mode_override_and_sums():
    manifests, modes = load_manifests(FIX), load_modes(FIX / "deploy/modes.yaml")
    assert budget(manifests["llm"], "A")["vram_mb"] == 4000 and budget(manifests["llm"], "B")["vram_mb"] == 7000
    assert sorted(residents(manifests, "C")) == ["kws"]
    names, vram, ram, _, ok = check_mode(manifests, modes, "A", "stand")
    assert (vram, ram, ok) == (4500, 1800, True)                    # KV reserve of A is added to VRAM
    names, vram, ram, _, ok = check_mode(manifests, modes, "B", "stand")
    assert (vram, ram, ok) == (7000, 1700, True)


def test_orin_counts_one_pool():
    """A fits the stand (VRAM 4500 <= 8000, RAM 1800 <= 4000) but not Orin's single pool: 4500 + 1800 > 6000."""
    manifests, modes = load_manifests(FIX), load_modes(FIX / "deploy/modes.yaml")
    assert check_mode(manifests, modes, "A", "stand")[-1] is True
    assert check_mode(manifests, modes, "A", "orin")[-1] is False
    assert "mode A on orin" in errors(FIX, target="orin")


def test_no_never_evict(tree):
    edit(tree, "services/kws/manifest.yaml", lambda d: d.update(never_evict=False))
    assert "exactly one" in errors(tree)


def test_two_never_evict(tree):
    edit(tree, "services/voice/manifest.yaml", lambda d: d.update(never_evict=True))
    assert "exactly one" in errors(tree)


def test_never_evict_must_be_resident_everywhere(tree):
    edit(tree, "services/kws/manifest.yaml", lambda d: d["residency"].update(C="evacuated"))
    assert "resident in every mode" in errors(tree)


def test_measured_on_required(tree):
    edit(tree, "services/voice/manifest.yaml", lambda d: d["budget"].pop("measured_on"))
    assert "measured_on is required" in errors(tree)


def test_vram_over_budget(tree):
    edit(tree, "services/llm/manifest.yaml", lambda d: d["modes"]["B"]["budget"].update(vram_mb=8500))
    msg = errors(tree)
    assert "mode B on stand" in msg and "over the budget" in msg


def test_ram_over_budget(tree):
    edit(tree, "services/voice/manifest.yaml", lambda d: d["budget"].update(ram_mb=3500))
    assert "mode A on stand" in errors(tree)


def test_kv_reserve_counts(tree):
    edit(tree, "deploy/modes.yaml", lambda d: d["modes"]["A"].update(kv_reserve_mb=4500))
    assert "mode A on stand" in errors(tree)


def test_unknown_lifecycle_and_residency(tree):
    edit(tree, "services/voice/manifest.yaml", lambda d: (d["lifecycle"].update(freeze="pause"),
                                                           d["residency"].update(B="maybe")))
    msg = errors(tree)
    assert "lifecycle.freeze 'pause'" in msg and "residency.B" in msg


def test_slot_lifecycle_needs_healthcheck(tree):
    edit(tree, "services/llm/manifest.yaml", lambda d: d.update(healthcheck=None))
    assert "needs a healthcheck" in errors(tree)


def test_all_problems_listed_at_once(tree):
    edit(tree, "services/kws/manifest.yaml", lambda d: d.update(never_evict=False))
    edit(tree, "services/voice/manifest.yaml", lambda d: d["budget"].pop("measured_on"))
    msg = errors(tree)
    assert "exactly one" in msg and "measured_on" in msg
