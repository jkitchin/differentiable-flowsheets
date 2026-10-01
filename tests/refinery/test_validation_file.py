"""The validation reference file is present and complete (issue #298).

Kept out of ``test_validation.py`` because that module is ``release`` as a
whole, and a module-level mark cannot be taken off one class: a deleted or
truncated ``cdu_reference.json`` is wiring, and should fail on every commit.
"""

import json
from pathlib import Path

REF_DIR = Path(__file__).parent / "reference"
REF = json.loads((REF_DIR / "cdu_reference.json").read_text())


def test_it_records_where_it_came_from():
    p = REF["provenance"]
    for key in ("generated", "script", "idaes", "pyomo", "ipopt", "property_methods",
                "reference_simulator", "difflow_commit"):
        assert p[key], key
    assert "DWSIM" in p["reference_simulator"]
    assert (REF_DIR / "generate.py").exists()


def test_every_layer_is_there():
    for key in ("layer1", "layer2", "layer3", "layer4"):
        assert REF[key], key
    assert REF["layer3"]["solve"]["status"] == "optimal"
