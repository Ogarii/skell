"""The learned joint detector, numpy side: mirroring, decoding heatmaps and aligning sides."""
import numpy as np
import pytest

from sandbox.sketch_detect import PLACED
from sandbox.synth import MIRROR, SIDE_PAIRS, generate, mirror
from sandbox.joint_net import align_sides, decode


@pytest.fixture(scope="module")
def few():
    return generate(8, seed=11, size=192)


def test_mirror_swaps_sides_and_keeps_the_middle():
    for i, j in enumerate(PLACED):
        k = PLACED[MIRROR[i]]
        if j.startswith("l_"):
            assert k == "r_" + j[2:]
        elif j.startswith("r_"):
            assert k == "l_" + j[2:]
        else:
            assert k == j
    assert sorted(MIRROR) == list(range(len(PLACED)))


def test_mirror_twice_is_the_original(few):
    imgs, joints, _, _ = few
    im2, J2 = mirror(*mirror(imgs, joints))
    assert np.array_equal(im2, imgs)
    assert np.allclose(J2, joints, atol=1e-5)


def test_mirrored_labels_sit_on_the_same_ink(few):
    """The ink under each mirrored label equals the ink under the original joint it came from."""
    imgs, joints, _, _ = few
    mi, mj = mirror(imgs, joints)
    for n in range(len(imgs)):
        for i in range(len(PLACED)):
            c, r = np.floor(joints[n, i]).astype(int)
            c2, r2 = np.floor(mj[n, MIRROR[i]]).astype(int)
            if 0 <= c < 192 and 0 <= r < 192 and 0 <= c2 < 192:
                assert mi[n, r2, c2] == imgs[n, r, c]


def test_decode_finds_a_peak_with_subpixel_accuracy():
    h = w = 48
    true = np.array([[50.3, 101.7], [10.0, 180.0]])
    yy, xx = np.mgrid[0:h, 0:w]
    logits = np.stack([-((((xx + 0.5) * 4 - x) ** 2 + ((yy + 0.5) * 4 - y) ** 2) / (2 * 5.0 ** 2)) for x, y in true])
    J, conf = decode(logits)
    assert np.abs(J - true).max() < 0.5
    assert (conf > 0.8).all()


def test_align_sides_swaps_only_when_closer():
    ref = np.zeros((11, 2))
    J = np.zeros((11, 2))
    (la, ra), (ll, rl) = SIDE_PAIRS
    ref[la], ref[ra] = 10, 90
    J[la], J[ra] = 88, 12                      # arms the other way round
    ref[ll], ref[rl] = 30, 70
    J[ll], J[rl] = 31, 69                      # legs already matching
    A = align_sides(ref, J)
    assert np.allclose(A[la], 12) and np.allclose(A[ra], 88)
    assert np.allclose(A[ll], 31) and np.allclose(A[rl], 69)
