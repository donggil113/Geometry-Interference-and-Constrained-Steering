import pytest
import torch

import n3  # noqa: F401
from n3.methods import METHODS, Editor, EditSpec
from n3.saes import load_sae

torch.manual_seed(0)


@pytest.fixture(scope="module")
def acts():
    from n3.data import owt_texts
    from n3.models import TLModel

    m = TLModel("gpt2", hook="blocks.6.hook_resid_pre")
    toks = m.to_tokens(owt_texts("validation")[:4], max_len=48)
    H = m.resid(toks)
    return H[:, 1:].reshape(-1, 768).float()[:96]


@pytest.fixture(scope="module")
def relu_sae():
    w = load_sae("gpt2-small-res-jb", "blocks.6.hook_resid_pre")[0]
    w.center_input = True  # as in n3.settings for mean-invariant GPT-2
    return w


@pytest.fixture(scope="module")
def topk_sae():
    w = load_sae("gpt2-small-resid-post-v5-32k", "blocks.5.hook_resid_post")[0]
    w.center_input = True
    return w


def spec(method, T, alpha=40.0, **kw):
    T = torch.tensor(T)
    return EditSpec(method=method, T=T, wT=torch.ones(len(T)) / len(T), alpha=alpha,
                    generic_dir=torch.randn(768), feat_scale=None, **kw)


@pytest.mark.parametrize("method", METHODS)
def test_norm_matched(method, acts, relu_sae):
    ed = Editor(relu_sae, spec(method, [100, 2000]))
    delta = ed(acts)
    nrm = delta.norm(dim=-1)
    target = 0.0 if method == "none" else 40.0
    assert torch.allclose(nrm, torch.full_like(nrm, target), atol=1e-3), (method, nrm[:5])


def test_pinv_preserves_active_protected_exactly(acts, relu_sae):
    sae = relu_sae
    T = [100]
    ed = Editor(sae, spec("pinv", T, alpha=20.0))
    delta = ed(acts)
    f0 = sae.encode(acts)
    f1 = sae.encode(acts + delta)
    P = f0 > 0
    P[:, T] = False
    rel = ((f1 - f0) * P).norm(dim=-1) / (f0 * P).norm(dim=-1)
    assert rel.max() < 1e-3
    # target pre-activation increases
    assert ((sae.pre(acts + delta) - sae.pre(acts))[:, T] > 0).all()


def test_dec_proj_is_close_to_decoder(acts, relu_sae):
    sae = relu_sae
    ed = Editor(sae, spec("dec_proj", [100], alpha=1.0))
    delta = ed(acts)
    cos = (delta @ sae.W_dec[100]) / sae.W_dec[100].norm()
    assert (cos > 0.3).all()
    J = sae.W_enc.T
    f0 = sae.encode(acts)
    P = f0 > 0
    P[:, 100] = False
    moved = (delta @ sae.W_enc) * P
    assert moved.abs().max() < 1e-3


def test_finite_step_reduces_new_activations(acts, relu_sae):
    sae = relu_sae
    res = {}
    for m in ["pinv", "pinv_fs"]:
        ed = Editor(sae, spec(m, [100], alpha=60.0, fs_rounds=4))
        ed(acts)
        res[m] = ed.summary()["new_active_per_pos"]
    assert res["pinv_fs"] < res["pinv"]


def test_ln_jacobian_matches_autograd(acts, topk_sae):
    sae = topk_sae
    h = acts[:3].clone()
    idx = torch.tensor([[5, 77, 900], [1, 2, 3], [10, 11, -1]])
    J = sae.pre_jacobian(h, idx)
    for i in range(3):
        hi = h[i].clone().requires_grad_(True)
        pre = sae.pre(hi.unsqueeze(0))[0]
        for a in range(3):
            j = int(idx[i, a])
            if j < 0:
                assert J[i, a].abs().max() == 0
                continue
            g, = torch.autograd.grad(pre[j], hi, retain_graph=True)
            assert torch.allclose(g, J[i, a], atol=1e-5, rtol=1e-3), (i, a, (g - J[i, a]).abs().max())


def test_enc_is_centred_encoder_row_for_relu(acts, relu_sae):
    ed = Editor(relu_sae, spec("enc", [100], alpha=1.0))
    delta = ed(acts[:4])
    w = relu_sae.W_enc[:, 100]
    w = w - w.mean()
    assert torch.allclose(delta, (w / w.norm()).expand(4, -1), atol=1e-5)


@pytest.mark.parametrize("method", [m for m in METHODS if m != "none"])
def test_deltas_are_mean_free_under_centering(method, acts, relu_sae):
    delta = Editor(relu_sae, spec(method, [100, 2000], alpha=25.0))(acts[:16])
    assert delta.mean(-1).abs().max() < 1e-4


def test_centering_makes_sae_blind_to_mean(acts, relu_sae):
    h = acts[:8]
    assert torch.allclose(relu_sae.encode(h + 7.0), relu_sae.encode(h), atol=1e-4)


def test_opt_under_inference_mode(acts, relu_sae):
    ed = Editor(relu_sae, spec("opt", [100], alpha=30.0))
    with torch.inference_mode():
        delta = ed(acts[:8].clone())
    assert torch.allclose(delta.norm(dim=-1), torch.full((8,), 30.0), atol=1e-3)


def test_randP_matches_constraint_count(acts, relu_sae):
    ed_real = Editor(relu_sae, spec("dec_proj_fs", [100], alpha=30.0))
    ed_real(acts)
    ed_rand = Editor(relu_sae, spec("dec_proj_fs_randP", [100], alpha=30.0))
    d = ed_rand(acts)
    assert torch.allclose(d.norm(dim=-1), torch.full((acts.shape[0],), 30.0), atol=1e-3)
    assert abs(ed_real.summary()["n_constraints"] - ed_rand.summary()["n_constraints"]) < 1e-6
    # random protected rows do not protect the actually-active features
    assert ed_rand.summary()["p_rel_change"] > ed_real.summary()["p_rel_change"]
