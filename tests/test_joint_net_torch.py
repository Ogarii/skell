"""Training side of the joint model (ml/): numpy and PyTorch agree, the augmentation keeps labels on the ink,
the side-swap loss, and a tiny training run end to end. Skipped where PyTorch isn't installed."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sandbox.sketch_detect import PLACED  # noqa: E402
from sandbox.synth import SIDE_PAIRS, generate  # noqa: E402


@pytest.fixture(scope="module")
def few():
    return generate(8, seed=11, size=192)



def test_numpy_net_matches_torch(tmp_path, few):
    from ml.model import JointNet, export_numpy, to_ink
    from sandbox.joint_net import JointNet as NpNet
    torch.manual_seed(0)
    m = JointNet()
    with torch.no_grad():                      # non-trivial batch-norm statistics, so the folding is tested
        for mod in m.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.running_mean.uniform_(-0.2, 0.2)
                mod.running_var.uniform_(0.5, 2.0)
                mod.weight.uniform_(0.5, 1.5)
                mod.bias.uniform_(-0.2, 0.2)
    m.eval()
    export_numpy(m, tmp_path / "w.npz")
    img = few[0][0]
    with torch.no_grad():
        ref = m(to_ink(torch.as_tensor(img[None]))).numpy()[0]
    got = NpNet.load(tmp_path / "w.npz").logits(img)
    assert got.shape == ref.shape == (11, 48, 48)
    assert np.abs(got - ref).max() < 1e-3 * max(1.0, np.abs(ref).max())


def test_augmentation_keeps_labels_on_the_ink(few):
    from ml.model import to_ink
    from ml.train_joints import augment
    imgs, joints, _, _ = few
    gen = torch.Generator().manual_seed(3)
    x, J = augment(to_ink(torch.as_tensor(imgs)), torch.as_tensor(joints), gen)
    x, J = x[:, 0].numpy(), J.numpy()
    # every joint except the head (the circle's centre is blank paper) sits on ink in the augmented image
    hits = []
    for n in range(len(imgs)):
        for i, j in enumerate(PLACED):
            if j == "head":
                continue
            c, r = np.floor(J[n, i]).astype(int)
            if 2 <= c < 190 and 2 <= r < 190:
                hits.append(x[n, r - 2:r + 3, c - 2:c + 3].max() > 0.15)
    assert np.mean(hits) > 0.95


def test_flip_invariant_loss_ignores_swapped_arms(few):
    from ml.train_joints import loss_fn, targets
    _, joints, _, _ = few
    J = torch.as_tensor(joints)
    logits = torch.log(targets(J, 48, 48, 5.0).clamp_min(1e-12))
    Js = J.clone()
    (la, ra), _ = SIDE_PAIRS
    Js[:, la + ra] = J[:, ra + la]
    assert abs(float(loss_fn(logits, J, 5.0)) - float(loss_fn(logits, Js, 5.0))) < 1e-4


def test_smoke_training_run(tmp_path):
    from ml.train_joints import main
    main(["--smoke", "--out", str(tmp_path / "run"), "--cache", str(tmp_path / "cache"), "--real", "none"])
    assert (tmp_path / "run" / "joint_net.npz").is_file()
    assert (tmp_path / "run" / "report.json").is_file()
