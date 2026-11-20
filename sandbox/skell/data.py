"""Load the shared rule data in data/ and the golden fixtures in fixtures/.

Rules live in the JSON files, never in code. This module only reads and checks their structure.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
FIXTURE_DIR = ROOT / "fixtures"


def load_json(name: str, folder: Path = DATA_DIR) -> dict:
    with open(folder / name, encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class Bone:
    id: str
    from_joint: str
    to_joint: str
    length: float
    kind: str                 # "posable" or "rigid"
    parent_bone: str | None
    rest_axis: np.ndarray     # unit vector, bone direction in the rest pose, in the parent frame
    limit: dict | None
    label: str


@dataclass(frozen=True)
class Skeleton:
    format: str
    name: str
    root: str
    joints: list[str]
    bones: list[Bone]         # parent before child
    head_radius: float
    standing_height: float
    symmetric: bool

    @classmethod
    def from_dict(cls, raw: dict) -> "Skeleton":
        bones = [
            Bone(
                id=b["id"], from_joint=b["from"], to_joint=b["to"], length=float(b["length"]),
                kind=b["kind"], parent_bone=b["parent_bone"],
                rest_axis=np.asarray(b["rest_axis"], dtype=float), limit=b["limit"], label=b.get("label", ""),
            )
            for b in raw["bones"]
        ]
        skel = cls(
            format=raw["format"], name=raw["name"], root=raw["root"], joints=list(raw["joints"]),
            bones=bones, head_radius=float(raw["head_radius"]),
            standing_height=float(raw["standing_height"]), symmetric=bool(raw["symmetric"]),
        )
        skel._validate()
        return skel

    def _validate(self) -> None:
        if self.root not in self.joints:
            raise ValueError(f"root {self.root!r} is not a joint")
        reached = {self.root}
        ending_at = {}  # joint -> bone id that ends there
        for b in self.bones:
            if b.id != f"{b.from_joint}>{b.to_joint}":
                raise ValueError(f"bone {b.id!r} must be named '{b.from_joint}>{b.to_joint}'")
            if b.kind not in ("posable", "rigid"):
                raise ValueError(f"bone {b.id}: unknown kind {b.kind!r}")
            if b.from_joint not in reached:
                raise ValueError(f"bone {b.id}: joint {b.from_joint!r} used before a bone reaches it")
            if b.to_joint in reached:
                raise ValueError(f"bone {b.id}: joint {b.to_joint!r} reached twice")
            expected_parent = ending_at.get(b.from_joint)  # None when it starts at the root
            if b.parent_bone != expected_parent:
                raise ValueError(f"bone {b.id}: parent_bone should be {expected_parent!r}, not {b.parent_bone!r}")
            if abs(float(np.linalg.norm(b.rest_axis)) - 1.0) > 1e-9:
                raise ValueError(f"bone {b.id}: rest_axis is not a unit vector")
            if b.length < 0:
                raise ValueError(f"bone {b.id}: negative length")
            reached.add(b.to_joint)
            ending_at[b.to_joint] = b.id
        if reached != set(self.joints):
            raise ValueError(f"joints not reached by any bone: {sorted(set(self.joints) - reached)}")

    def bone(self, bone_id: str) -> Bone:
        for b in self.bones:
            if b.id == bone_id:
                return b
        raise KeyError(bone_id)

    @property
    def posable_ids(self) -> list[str]:
        return [b.id for b in self.bones if b.kind == "posable"]

    @property
    def rigid_ids(self) -> list[str]:
        return [b.id for b in self.bones if b.kind == "rigid"]

    def children(self, bone_id: str) -> list[Bone]:
        return [b for b in self.bones if b.parent_bone == bone_id]


def load_skeleton(name: str = "skell_v1_reference_skeleton.json") -> Skeleton:
    return Skeleton.from_dict(load_json(name))


def load_rules() -> dict:
    return load_json("skell_rules_v1.json")


def load_reference_pose() -> dict:
    return load_json("reference_kneeling_pose.json")


def reference_frame() -> dict:
    return load_reference_pose()["options"][0]["frames"][0]


def pose_library() -> list[dict]:
    """Viewer poses: the reference kneeling pose first, then data/sandbox_poses_v1.json.
    Each entry has name, label and frame (spec 10.4 frame format)."""
    lib = [{"name": "reference_kneeling", "label": "Kneel, pick up box (ref)", "frame": reference_frame()}]
    return lib + load_json("sandbox_poses_v1.json")["poses"]


def load_intents() -> dict:
    return load_json("intent_vocab_v1.json")


def all_intent_tags() -> set[str]:
    vocab = load_intents()
    return {tag for key, tags in vocab.items() if key not in ("format", "notes") for tag in tags}


def load_camera_phrases() -> dict:
    return load_json("camera_phrases_v1.json")


def load_fixture(name: str = "expected_reference_kneeling.json") -> dict:
    return load_json(name, FIXTURE_DIR)
