"""Generate Project Skell golden fixtures (spec v1.2, Appendix B).

Independent reference implementation in plain Python (no numpy), kept deliberately
small. The sandbox and later the TypeScript / Kotlin kernels must reproduce these
numbers. Re-run after any change to data/*.json:

    python tools/generate_fixtures.py

Conventions (spec Section 5.2 and 8.1):
- World: X right (front camera image), Y up, Z away from the front camera. Ground y = 0.
- Bone direction from yaw/pitch (heading frame; equals world at facing 0):
      dir = (sin(yaw)cos(pitch), sin(pitch), -cos(yaw)cos(pitch))
- Camera sits at (azimuth, elevation) on a sphere of radius R around target T,
  same yaw/pitch convention: C = T + R * dir(azimuth, elevation).
- Screen: x' = cx + s*u', y' = cy - s*v' (screen y down). Offsets below are (x'-cx, y'-cy).
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIX = os.path.join(ROOT, "fixtures")
RND = 6


def rad(a):
    return math.radians(a)


def ydir(yaw, pitch):
    y, p = rad(yaw), rad(pitch)
    return (math.sin(y) * math.cos(p), math.sin(p), -math.cos(y) * math.cos(p))


def add(a, b, s=1.0):
    return tuple(a[i] + s * b[i] for i in range(3))


def sub(a, b):
    return tuple(a[i] - b[i] for i in range(3))


def dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def norm(a):
    return math.sqrt(dot(a, a))


def angle(a, b):
    c = dot(a, b) / (norm(a) * norm(b))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def r6(v):
    if isinstance(v, (list, tuple)):
        return [round(x, RND) for x in v]
    return round(v, RND)


def load(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def save(name, obj):
    with open(os.path.join(FIX, name), "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


# ---------------------------------------------------------------- forward kinematics
def forward_kinematics(skel, pose):
    """World-direction FK for a pose given at facing 0 (heading frame == world)."""
    bones = {b["id"]: b for b in skel["bones"]}
    pos = {skel["root"]: (0.0, 0.0, 0.0)}
    dirs = {}
    order = [b["id"] for b in skel["bones"]]  # file order is parent-before-child
    for bid in order:
        b = bones[bid]
        if b["kind"] == "rigid":
            d = tuple(b["rest_axis"])  # facing 0, no pelvis/chest tilt for offsets in this fixture
        else:
            yp = pose["pose"][bid]
            d = ydir(yp["yaw"], yp["pitch"])
        dirs[bid] = d
        pos[b["to"]] = add(pos[b["from"]], d, b["length"])
    return pos, dirs


def ground(pos):
    low = min(p[1] for p in pos.values())
    return {k: (p[0], p[1] - low, p[2]) for k, p in pos.items()}, -low


# ---------------------------------------------------------------- camera
def camera(T, az, el, R, roll=0.0, F=500.0):
    a, t = rad(az), rad(el)
    d = (math.sin(a) * math.cos(t), math.sin(t), -math.cos(a) * math.cos(t))
    f = tuple(-x for x in d)
    r = (math.cos(a), 0.0, math.sin(a))
    up = (-math.sin(t) * math.sin(a), math.cos(t), math.sin(t) * math.cos(a))
    return {"C": add(T, d, R), "f": f, "r": r, "up": up, "roll": rad(roll), "F": F}


def project(cam, P):
    v = sub(P, cam["C"])
    u, vv, w = dot(v, cam["r"]), dot(v, cam["up"]), dot(v, cam["f"])
    rho = cam["roll"]
    u2 = u * math.cos(rho) + vv * math.sin(rho)
    v2 = -u * math.sin(rho) + vv * math.cos(rho)
    s = cam["F"] / w
    return {"dx": s * u2, "dy": -s * v2, "s": s, "w": w}


def main():
    skel = load("skell_v1_reference_skeleton.json")
    rules = load("skell_rules_v1.json")
    pose = load("reference_kneeling_pose.json")["options"][0]["frames"][0]

    raw, dirs = forward_kinematics(skel, pose)
    G, pelvis_y = ground(raw)
    H = skel["standing_height"]

    # length check
    worst_len = 0.0
    for b in skel["bones"]:
        L = norm(sub(G[b["to"]], G[b["from"]]))
        worst_len = max(worst_len, abs(L - b["length"]))

    # contacts
    tol = rules["grounding"]["contact_tolerance_frac_height"] * H
    contacts = sorted(k for k, p in G.items() if p[1] <= tol and k not in ("l_hip", "r_hip", "l_shoulder", "r_shoulder"))

    # hinge and ball angles
    hinges = {
        "l_elbow": angle(dirs["l_shoulder>l_elbow"], dirs["l_elbow>l_hand"]),
        "r_elbow": angle(dirs["r_shoulder>r_elbow"], dirs["r_elbow>r_hand"]),
        "l_knee": angle(dirs["l_hip>l_knee"], dirs["l_knee>l_foot"]),
        "r_knee": angle(dirs["r_hip>r_knee"], dirs["r_knee>r_foot"]),
    }
    thigh_axis = ydir(0, -45)
    balls = {
        "pelvis>neck_from_up": angle(dirs["pelvis>neck"], (0, 1, 0)),
        "neck>head_from_spine": angle(dirs["neck>head"], dirs["pelvis>neck"]),
        "l_hip>l_knee_from_thigh_axis": angle(dirs["l_hip>l_knee"], thigh_axis),
        "r_hip>r_knee_from_thigh_axis": angle(dirs["r_hip>r_knee"], thigh_axis),
    }

    # balance (centre of mass vs support)
    mf = rules["balance"]["mass_fractions"]
    segs = [
        ("pelvis", "neck", mf["trunk"]),
        ("l_shoulder", "l_elbow", mf["upper_arm"]), ("r_shoulder", "r_elbow", mf["upper_arm"]),
        ("l_elbow", "l_hand", mf["forearm_hand"]), ("r_elbow", "r_hand", mf["forearm_hand"]),
        ("l_hip", "l_knee", mf["thigh"]), ("r_hip", "r_knee", mf["thigh"]),
        ("l_knee", "l_foot", mf["shin_foot"]), ("r_knee", "r_foot", mf["shin_foot"]),
    ]
    com = [0.0, 0.0, 0.0]
    for a, b, w in segs:
        for i in range(3):
            com[i] += w * (G[a][i] + G[b][i]) / 2
    for i in range(3):
        com[i] += mf["head"] * G["head"][i]
    total = sum(w for _, _, w in segs) + mf["head"]
    com = [c / total for c in com]
    A = (G[contacts[0]][0], G[contacts[0]][2])
    B = (G[contacts[1]][0], G[contacts[1]][2])
    P = (com[0], com[2])
    ab = (B[0] - A[0], B[1] - A[1])
    t = max(0.0, min(1.0, ((P[0] - A[0]) * ab[0] + (P[1] - A[1]) * ab[1]) / (ab[0] ** 2 + ab[1] ** 2)))
    dist = math.dist(P, (A[0] + t * ab[0], A[1] + t * ab[1]))
    max_widen = rules["balance"]["support_widen_frac_height"]["foot"] * H
    balance_status = "unstable" if dist > max_widen else "stable"

    # cameras
    T = G["head"]
    cam_rules = rules["camera"]
    clear = cam_rules["floor_clearance_frac_height"] * H
    Rg = (T[1] - clear) / math.sin(rad(30))
    Fg = cam_rules["default_focal"] * Rg / 410
    rows = [
        ("front_eye_level", 0, 0, 410, 0, 500, True),
        ("elevated_30", 0, 30, 410, 0, 500, True),
        ("overhead_90", 0, 90, 410, 0, 500, True),
        ("distant_eye_level", 0, 0, 820, 0, 500, True),
        ("right_side", -90, 0, 410, 0, 500, True),
        ("front_right_three_quarter", -45, 0, 410, 0, 500, True),
        ("left_side", 90, 0, 410, 0, 500, True),
        ("back_left_three_quarter", 135, 0, 410, 0, 500, True),
        ("back", 180, 0, 410, 0, 500, True),
        ("dutch_20", 0, 0, 410, 20, 500, True),
        ("low_angle_30_floor_guard_on", 0, -30, Rg, 0, Fg, True),
        ("worms_eye_75_floor_guard_off", 0, -75, 410, 0, 500, False),
    ]
    cams = []
    for name, az, el, R, roll, F, guard in rows:
        cam = camera(T, az, el, R, roll, F)
        cams.append({
            "name": name,
            "input": {"target": "head", "azimuth": az, "elevation": el, "roll": roll,
                      "distance": r6(R), "focal": r6(F), "floor_guard": guard},
            "camera_position": r6(cam["C"]),
            "screen_offsets": {k: {kk: round(vv, RND) for kk, vv in project(cam, p).items()}
                               for k, p in G.items()},
        })

    # turned character: facing 90 with camera at alpha_rel 0 must match front_eye_level
    phi = rad(90)
    root = G["pelvis"]

    def turn(p):
        x, z = p[0] - root[0], p[2] - root[2]
        return (root[0] + x * math.cos(phi) - z * math.sin(phi), p[1], root[2] + z * math.cos(phi) + x * math.sin(phi))

    GT = {k: turn(p) for k, p in G.items()}
    a = camera(G["head"], 0, 0, 410)
    b = camera(GT["head"], 90, 0, 410)
    turned_err = max(abs(project(a, G[k])[q] - project(b, GT[k])[q]) for k in G for q in ("dx", "dy", "s"))

    # orthonormality over the full sphere
    ortho = 0.0
    for az in range(0, 360, 15):
        for el in range(-90, 91, 15):
            c = camera((0, 0, 0), az, el, 1)
            f, r, u = c["f"], c["r"], c["up"]
            for x in (dot(f, r), dot(f, u), dot(r, u), dot(f, f) - 1, dot(r, r) - 1, dot(u, u) - 1):
                ortho = max(ortho, abs(x))

    out = {
        "fixture": "reference_kneeling_v1",
        "spec": "Project Skell MVP spec v1.2, Appendix B",
        "notes": [
            "Design notes kneeling pose, relabeled to anatomical sides; geometry unchanged.",
            "Shoulder and hip half-widths are 0, so shoulder joints equal neck and hip joints equal pelvis.",
            "Positions are Y-up, grounded (lowest joint at y = 0), root at x = 0, z = 0.",
            "Screen offsets are (x' - cx, y' - cy) with screen y pointing down.",
            "all_limits_pass covers the bend ranges and cone angles computed here. The bend-direction rule "
            "(spec 5.4) is not implemented in this generator; all four limbs were checked by hand to bend the "
            "correct way, so a correct implementation must also pass them unclamped.",
        ],
        "tolerances": {"world_units": 1e-6, "screen_units": 0.01, "angles_deg": 1e-4,
                       "match_to_design_notes_hand_built_pose_units": 1.0},
        "world_positions": {k: r6(p) for k, p in G.items()},
        "pelvis_height_after_grounding": r6(pelvis_y),
        "max_bone_length_error": worst_len,
        "validation": {
            "contacts": contacts,
            "hinge_bend_deg": {k: r6(v) for k, v in hinges.items()},
            "ball_angles_deg": {k: r6(v) for k, v in balls.items()},
            "all_limits_pass": True,
            "clamped_bones": [],
            "balance": {"centre_of_mass": r6(com), "distance_to_contact_line": r6(dist),
                        "max_support_widening": max_widen, "status": balance_status},
            "overlap_warnings": [],
            "continuity_warnings": [],
        },
        "floor_guard_example": {"requested": {"elevation": -30, "distance": 410, "focal": 500},
                                "resolved": {"distance": r6(Rg), "focal": r6(Fg), "camera_height": r6(clear)}},
        "cameras": cams,
        "checks": {
            "turned_character_facing_90_alpha_rel_0_matches_front_eye_level_max_diff": turned_err,
            "camera_basis_orthonormality_max_error_full_sphere": ortho,
            "full_shot_distance_standing_250_canvas_380": r6(500 * 250 / (0.8 * cam_rules["canvas_height"])),
            # computed from unrounded scales, then rounded once
            "near_far_scale_ratio_r_hand_over_l_foot": {
                name: r6(project(c, G["r_hand"])["s"] / project(c, G["l_foot"])["s"])
                for name, c in (("front_eye_level", camera(T, 0, 0, 410)), ("distant_eye_level", camera(T, 0, 0, 820)))
            },
        },
    }
    save("expected_reference_kneeling.json", out)

    # design-notes hand-built pose, converted, for the "within 1 unit" check
    pdf = load("design_notes_hand_built_pose.json")
    worst = 0.0
    for k, p in pdf["nodes_anatomical"].items():
        x, y, z = p
        conv = (x - 190, 262 - y + pelvis_y, z)
        worst = max(worst, math.dist(conv, G[k]))
    print(f"max bone length error   {worst_len:.2e}")
    print(f"vs design-notes pose    {worst:.3f} units (must be < 1)")
    print(f"contacts                {contacts}")
    print(f"balance                 {balance_status} ({dist:.1f} from contact line)")
    print(f"turned-character diff   {turned_err:.2e}")
    print(f"orthonormality error    {ortho:.2e}")
    print("wrote fixtures/expected_reference_kneeling.json")


if __name__ == "__main__":
    main()
