import torch

def _vorticity_to_velocity(w, L=2 * torch.pi):
    _, _, H, W = w.shape
    device, dtype = w.device, w.dtype

    kx = torch.fft.fftfreq(W, d=L / W, device=device) * 2 * torch.pi
    ky = torch.fft.fftfreq(H, d=L / H, device=device) * 2 * torch.pi
    KX, KY = torch.meshgrid(kx, ky, indexing="xy")
    K2 = KX ** 2 + KY ** 2
    K2[0, 0] = 1.0

    w_h = torch.fft.fftn(w, dim=[2, 3])
    psi_h = w_h / K2

    u_h = 1j * KY * psi_h
    v_h = -1j * KX * psi_h

    u = torch.fft.ifftn(u_h, dim=[2, 3]).real
    v = torch.fft.ifftn(v_h, dim=[2, 3]).real
    return u, v

def _rmse(pred, target):
    rmse = torch.sqrt(torch.mean((pred - target) ** 2))
    return rmse

def _r2(pred, target, eps=1e-12):
    ybar = target.mean(0, keepdim=True)
    ss_res = torch.sum((pred - target) ** 2)
    ss_tot = torch.sum((target - ybar) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + eps)
    return r2

def _frmse(pred, target):
    pred_F = torch.fft.fftn(pred, dim=[2, 3])
    target_F = torch.fft.fftn(target, dim=[2, 3])
    frmse = torch.sqrt(torch.mean((pred_F.abs() - target_F.abs()) ** 2))
    return frmse

def _ke(pred, target):
    pu, pv = _vorticity_to_velocity(pred)
    tu, tv = _vorticity_to_velocity(target)
    pred_k = 0.5 * (pu ** 2 + pv ** 2)
    target_k = 0.5 * (tu ** 2 + tv ** 2)
    return (pred_k - target_k).abs().mean()

def eval_metrics(pred, target):
    rmse = _rmse(pred, target)
    r2 = _r2(pred, target)
    frmse = _frmse(pred, target)
    ke = _ke(pred, target)

    return {
        "rmse": rmse.item(),
        "r2": r2.item(),
        "frmse": frmse.item(),
        "ke": ke.item(),
    }