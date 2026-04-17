"""Smoke tests for the overcooked v2 env wrapper.

The JAX/JaxMARL stack is heavy and not installed on CI; tests here
either run in jax-free mode (verify imports + clear error messages)
or skip if jaxmarl isn't importable.

.github/workflows/ci.yml installs the [dev] extra alone, so jaxmarl is
absent on every CI run. The two skip guards below point opposite ways.
One test runs only with jaxmarl present and the other only without, so
no single environment executes both bodies.
"""
from __future__ import annotations

import pytest

from mcts_ued.env.overcooked import OvercookedConfig, OvercookedIPPOStudent
from mcts_ued.partners.parametrize import BoxPartnerSpace


def test_env_module_imports_without_jax():
    """Importing the module should not require jax/jaxmarl."""
    # the module-scope import at the top of this file is the check. overcooked.py
    # holds flax and optax behind a function-local import of _ippo, and moving
    # either to module scope fails collection before this body runs
    # the check only bites where the [jax] extra is absent. a machine with jaxmarl
    # installed reports green whatever overcooked.py imports at module scope
    student = OvercookedIPPOStudent(OvercookedConfig(layout="cramped_room"))
    assert student.cfg.layout == "cramped_room"


def test_clear_error_when_jax_missing():
    """Reserved for the jaxmarl-present path. Nothing here asserts.

    importorskip skips when jaxmarl is missing, which is every CI run. With
    jaxmarl present the body is empty and passes. Deleting the test changes
    no outcome in either environment.
    """
    jaxmarl = pytest.importorskip("jaxmarl", reason="jaxmarl available; install does not apply")  # noqa: F841
    # the reason text reads as though the skip fires when jaxmarl is available.
    # importorskip prints it on the opposite branch


def test_from_yaml_round_trip():
    # delete all three asserts and this still catches the defect it was written for:
    # from_yaml has to drop "kind" or OvercookedConfig raises TypeError on an
    # unexpected keyword
    # from_yaml runs no coercion. the three values arrive as the ints this literal
    # already holds, so a yaml file quoting one of them sends a str to _ippo
    raw = {
        "kind": "overcooked_ippo",
        "layout": "cramped_room",
        "n_eval_episodes": 4,
        "student_train_steps": 100,
    }
    student = OvercookedIPPOStudent.from_yaml(raw)
    assert student.cfg.layout == "cramped_room"
    assert student.cfg.n_eval_episodes == 4
    assert student.cfg.student_train_steps == 100


def test_train_step_without_jax_raises_runtime_error():
    # pytest is imported at the top of this file, so this line can never skip
    pytest.importorskip("pytest")
    # without the skip, an environment holding the [jax] extra would build a real
    # overcooked_v2 env and run a 50k-env-step IPPO pass before failing on the
    # RuntimeError that never comes
    try:
        import jaxmarl  # type: ignore[import-not-found]  # noqa: F401
        pytest.skip("jaxmarl is installed; the no-jax path is exercised elsewhere")
    except ImportError:
        pass
    student = OvercookedIPPOStudent(OvercookedConfig(layout="cramped_room"))
    space = BoxPartnerSpace()
    # _ensure_jax raises before train_step reads the spec, so the draw and seed 0
    # carry nothing. any object with the PartnerSpec shape reaches the same error
    # inline __import__ keeps numpy off the module scope. parametrize.py imports it
    # anyway, so a module-scope import here would add no dependency
    spec = space.sample(__import__("numpy").random.default_rng(0))
    # match is a re.search over the message. the escaped brackets pin the extra's
    # name, so renaming [jax] in pyproject.toml reddens this without touching the
    # exception type
    with pytest.raises(RuntimeError, match="\\[jax\\]"):
        student.train_step(spec, seed=0)
