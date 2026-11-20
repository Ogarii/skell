"""Guards the golden fixtures: regenerating them must reproduce the committed file."""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIX = ROOT / "fixtures" / "expected_reference_kneeling.json"


def test_fixtures_are_current(tmp_path):
    before = json.loads(FIX.read_text())
    subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "generate_fixtures.py")], check=True, cwd=ROOT)
    after = json.loads(FIX.read_text())
    assert before == after, "fixtures/ is out of date with data/; review the diff and commit it"


def test_reference_checks():
    f = json.loads(FIX.read_text())
    assert f["max_bone_length_error"] < 1e-9
    assert f["validation"]["contacts"] == ["l_knee", "r_foot"]
    assert f["validation"]["balance"]["status"] == "unstable"
    assert f["checks"]["camera_basis_orthonormality_max_error_full_sphere"] < 1e-12
    assert f["checks"]["turned_character_facing_90_alpha_rel_0_matches_front_eye_level_max_diff"] < 1e-9
