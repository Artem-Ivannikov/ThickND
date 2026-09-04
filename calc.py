import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import fftconvolve
import mrcfile
from typing import List, Tuple, Optional

def load_mrc(path: str) -> Tuple[np.ndarray, float]:
    """Load an MRC file and return (data, pixel_size Å)."""
    with mrcfile.open(path, permissive=True) as mrc:
        data = mrc.data.copy()
        pix = float(mrc.voxel_size.x)
    return data, pix

def gaussian_quad(x: np.ndarray,
                  m1: float, s1: float, a1: float,
                  m2: float, s2: float, a2: float,
                  m3: float, s3: float, a3: float,
                  m4: float, s4: float, a4: float,
                  b: float) -> np.ndarray:
    """Sum of 4 Gaussians plus constant bias."""
    g1 = a1 * np.exp(-0.5 * ((x-m1)/s1)**2)
    g2 = a2 * np.exp(-0.5 * ((x-m2)/s2)**2)
    g3 = a3 * np.exp(-0.5 * ((x-m3)/s3)**2)
    g4 = a4 * np.exp(-0.5 * ((x-m4)/s4)**2)
    return g1 + g2 + g3 + g4 + b

def gaussian_quad_jac(x: np.ndarray,
                      m1: float, s1: float, a1: float,
                      m2: float, s2: float, a2: float,
                      m3: float, s3: float, a3: float,
                      m4: float, s4: float, a4: float,
                      b: float) -> np.ndarray:
    """Analytical Jacobian of the 4‑Gaussian model, shape (N,13)."""
    x = x[:, None]  # (N,1)
    def grads(mi, si, ai):
        d = x - mi
        expv = np.exp(-0.5*(d/si)**2)
        dm = ai * expv * (d/si**2)
        ds = ai * expv * (d**2/si**3)
        da =        expv
        return dm, ds, da

    dm1, ds1, da1 = grads(m1, s1, a1)
    dm2, ds2, da2 = grads(m2, s2, a2)
    dm3, ds3, da3 = grads(m3, s3, a3)
    dm4, ds4, da4 = grads(m4, s4, a4)
    db = np.ones_like(x)

    return np.hstack([
        dm1, ds1, da1,
        dm2, ds2, da2,
        dm3, ds3, da3,
        dm4, ds4, da4,
        db
    ])  # (N,13)

def theta_to_phi(theta, pixel_size, i0):
    """
    Convert 13 raw theta parameters (absolute Å, unitless fractions) to
    13 phi parameters (pixel‑relative means, pixel sigmas, unitless amplitudes).

    Works for both torch.Tensor (batched) and np.ndarray (single vector).
    The last dimension must have size 13.

    Parameters
    ----------
    theta : torch.Tensor or np.ndarray
        Shape (..., 13). Parameter order:
        [m0_ang, d0_ang, d1_frac, d2_frac, s1_ang, s2_ang,
         fs3, fs4, A1, A2, fA3, fA4, bias]
    pixel_size : float
        Voxel size in Å.
    i0 : int
        Starting pixel index of the extracted profile segment.

    Returns
    -------
    phi : same type and shape as theta
        Last dimension size 13, order:
        [m1_px, sigma1_px, A1, m2_px, sigma2_px, A2,
         m3_px, sigma3_px, A3, m4_px, sigma4_px, A4, bias]
    """
    # Unpack parameters (works for both torch and numpy via indexing)
    m0_ang   = theta[..., 0]
    d0_ang   = theta[..., 1]
    d1_frac  = theta[..., 2]
    d2_frac  = theta[..., 3]
    s1_ang   = theta[..., 4]
    s2_ang   = theta[..., 5]
    fs3      = theta[..., 6]
    fs4      = theta[..., 7]
    A1       = theta[..., 8]
    A2       = theta[..., 9]
    fA3      = theta[..., 10]
    fA4      = theta[..., 11]
    bias     = theta[..., 12]

    # Convert distances and means to pixel units
    inv_px = 1.0 / float(pixel_size)
    m0_px = m0_ang * inv_px - float(i0)
    delta0_px = d0_ang * inv_px
    delta1_px = delta0_px * d1_frac
    delta2_px = delta0_px * d2_frac
    sigma1_px = s1_ang * inv_px
    sigma2_px = s2_ang * inv_px
    sigma3_px = sigma1_px * fs3
    sigma4_px = sigma2_px * fs4

    # Compute the four Gaussian means
    m1_px = m0_px - delta0_px
    m2_px = m0_px + delta0_px
    m3_px = m1_px + delta1_px
    m4_px = m2_px - delta2_px

    # Amplitudes of satellite Gaussians
    A3 = A1 * fA3
    A4 = A2 * fA4

    # Stack appropriately depending on input type
    if isinstance(theta, np.ndarray):
        return np.stack([
            m1_px, sigma1_px, A1,
            m2_px, sigma2_px, A2,
            m3_px, sigma3_px, A3,
            m4_px, sigma4_px, A4,
            bias
        ], axis=-1)
    else:  # torch.Tensor
        return torch.stack([
            m1_px, sigma1_px, A1,
            m2_px, sigma2_px, A2,
            m3_px, sigma3_px, A3,
            m4_px, sigma4_px, A4,
            bias
        ], dim=-1)

class BatchedQuadGaussianModel(nn.Module):
    def __init__(self, batch_size: int,
                 lb: np.ndarray, ub: np.ndarray,
                 i0: int, pixel_size: float,
                 device: str):
        super().__init__()
        self.device = device
        self.raw    = nn.Parameter(torch.zeros(batch_size, 13, dtype=torch.double, device=device))
        self.lb     = torch.tensor(lb, dtype=torch.double, device=device).view(1,13)
        self.ub     = torch.tensor(ub, dtype=torch.double, device=device).view(1,13)
        self.i0        = i0
        self.pixel_size = pixel_size

    def forward(self, x: torch.Tensor):
        # x: (B, L) in pixel indices
        theta = self.lb + (self.ub - self.lb) * torch.sigmoid(self.raw)  # (B,13)
        phi = theta_to_phi(theta, self.pixel_size, self.i0)              # (B,13)

        # Unpack phi parameters (all in pixel units or unitless)
        m1_px, sigma1_px, A1, m2_px, sigma2_px, A2, m3_px, sigma3_px, A3, m4_px, sigma4_px, A4, bias = (
            phi[..., 0], phi[..., 1], phi[..., 2],
            phi[..., 3], phi[..., 4], phi[..., 5],
            phi[..., 6], phi[..., 7], phi[..., 8],
            phi[..., 9], phi[..., 10], phi[..., 11],
            phi[..., 12]
        )

        # Compute the four Gaussians
        xp = x.unsqueeze(2)  # (B,L,1)
        def G(m, s, A):
            z = (xp - m.unsqueeze(-1).unsqueeze(-1)) / s.unsqueeze(-1).unsqueeze(-1)
            return A.unsqueeze(-1).unsqueeze(-1) * torch.exp(-0.5*z*z)

        g1 = G(m1_px, sigma1_px, A1)
        g2 = G(m2_px, sigma2_px, A2)
        g3 = G(m3_px, sigma3_px, A3)
        g4 = G(m4_px, sigma4_px, A4)
        return (g1 + g2 + g3 + g4 + bias.unsqueeze(-1).unsqueeze(-1)).squeeze(2)  # (B,L)


def fit_batch_profile_gpu(
    segments: np.ndarray,          # (B, L)
    init_abs: np.ndarray,          # (13,)
    lb_abs: np.ndarray,            # (13,)
    ub_abs: np.ndarray,            # (13,)
    i0: int,
    pixel_size: float,
    corr_length: float,            # in Å
    reg_lambda: float = 0.0,
    max_iter: int = 200
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit B profiles of length L in one batch on GPU using 4 Gaussians.
    Returns:
      params_batch: (B,13)     -- raw parameters (same as original)
      cov_batch:    (B,13,13)  -- covariance in phi-space (old params, pixels)
      ssr_batch:    (B,)
      dof_array:    (B,)       -- per-profile degrees of freedom (N_eff - df_eff)
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, L = segments.shape
    seg_t = torch.from_numpy(segments.astype(np.float64)).to(device)
    y_norm = seg_t / (seg_t.amax(dim=1, keepdim=True) + 1e-8)

    # build model with bounds (unchanged)
    model = BatchedQuadGaussianModel(B,
                                     lb_abs, ub_abs,
                                     i0=i0, pixel_size=pixel_size,
                                     device=device)
    # invert sigmoid to set raw so that θ(init) = init_abs
    p0 = (init_abs - lb_abs) / (ub_abs - lb_abs)
    eps = 1e-6
    p0 = np.clip(p0, eps, 1-eps)
    raw0 = np.log(p0/(1-p0))  # logit
    model.raw.data[:] = torch.from_numpy(raw0).to(device).unsqueeze(0).expand(B,13)

    optimizer = optim.LBFGS([model.raw], max_iter=max_iter)
    x_t = torch.arange(L, dtype=torch.double, device=device).unsqueeze(0).expand(B, L)

    # --- build phi_init (torch for closure) and phi_init_np for covariance R building ---
    with torch.no_grad():
        theta_init_t = torch.from_numpy(init_abs.astype(np.float64)).to(device=device, dtype=torch.double)
        phi_init_torch = theta_to_phi(theta_init_t.unsqueeze(0), pixel_size, i0).squeeze(0)  # (13,) torch
    #phi_init_np = theta_to_phi_np(init_abs)  # numpy (13,)

    # --- Gauss-Newton R builder (numpy), returns 13x13 ---
    def build_R_GN(phi: np.ndarray, reg_lambda_local: float) -> np.ndarray:
        R = np.zeros((13,13), dtype=float)
        if reg_lambda_local <= 0.0:
            return R
        eps_small = 1e-12

        def add_rank1(g):
            R[:] += (2.0 * reg_lambda_local) * np.outer(g, g)

        p = phi
        # f1: bias
        g = np.zeros(13); g[12] = 1.0; add_rank1(g)

        # f2: A1 + bias
        g = np.zeros(13); g[2] = 1.0; g[12] = 1.0; add_rank1(g)

        # f3: A2 + bias
        g = np.zeros(13); g[5] = 1.0; g[12] = 1.0; add_rank1(g)

        # f4: A1 - A2
        g = np.zeros(13); g[2] = 1.0; g[5] = -1.0; add_rank1(g)

        # f5: sigma1 - sigma2
        g = np.zeros(13); g[1] = 1.0; g[4] = -1.0; add_rank1(g)

        # f6: (m1 + m2)/2
        g = np.zeros(13); g[0] = 0.5; g[3] = 0.5; add_rank1(g)

        # f7: A3/A1  (phi[8] / phi[2])
        A1 = p[2]; A3 = p[8]
        A1s = A1 if abs(A1) > eps_small else eps_small
        g = np.zeros(13)
        g[8] = 1.0 / A1s
        g[2] = - A3 / (A1s**2)
        add_rank1(g)

        # f8: A4/A2  (phi[11] / phi[5])
        A2 = p[5]; A4 = p[11]
        A2s = A2 if abs(A2) > eps_small else eps_small
        g = np.zeros(13)
        g[11] = 1.0 / A2s
        g[5]  = - A4 / (A2s**2)
        add_rank1(g)

        # f9: sigma3/sigma1  (phi[7]/phi[1])
        s1 = p[1]; s3 = p[7]
        s1s = s1 if abs(s1) > eps_small else eps_small
        g = np.zeros(13)
        g[7] = 1.0 / s1s
        g[1] = - s3 / (s1s**2)
        add_rank1(g)

        # f10: sigma4/sigma2 (phi[10]/phi[4])
        s2 = p[4]; s4 = p[10]
        s2s = s2 if abs(s2) > eps_small else eps_small
        g = np.zeros(13)
        g[10] = 1.0 / s2s
        g[4]  = - s4 / (s2s**2)
        add_rank1(g)

        # f11: 2*(m3 - m1)/(m2 - m1)
        a = p[0]; b = p[3]; c = p[6]
        v = b - a
        vs = v if abs(v) > eps_small else eps_small
        g = np.zeros(13)
        g[0] = 2.0 * (c - b) / (vs**2)
        g[3] = -2.0 * (c - a) / (vs**2)
        g[6] = 2.0 / vs
        add_rank1(g)

        # f12: 2*(m2 - m4)/(m2 - m1)
        a = p[0]; b = p[3]; d = p[9]
        v = b - a
        vs = v if abs(v) > eps_small else eps_small
        g = np.zeros(13)
        g[0] = 2.0 * (b - d) / (vs**2)
        g[3] = 2.0 * (d - a) / (vs**2)
        g[9] = -2.0 / vs
        add_rank1(g)

        return R

    # --- LBFGS closure (PyTorch only) ---
    def closure():
        optimizer.zero_grad()
        y_pred = model(x_t)                                 # (B,L)
        loss = ((y_pred - y_norm)**2).sum()                # SSR

        if reg_lambda > 0:
            theta_batch = model.lb + (model.ub - model.lb) * torch.sigmoid(model.raw)  # (B,13)
            phi_batch = theta_to_phi(theta_batch, pixel_size, i0)
            eps_div = 1e-12

            pen1  = (phi_batch[:, 12] - phi_init_torch[12])**2
            pen2  = (phi_batch[:, 2] + phi_batch[:, 12] - (phi_init_torch[2] + phi_init_torch[12]))**2
            pen3  = (phi_batch[:, 5] + phi_batch[:, 12] - (phi_init_torch[5] + phi_init_torch[12]))**2
            pen4  = (phi_batch[:, 2] - phi_batch[:, 5])**2
            pen5  = (phi_batch[:, 1] - phi_batch[:, 4])**2
            pen6  = (((phi_batch[:, 0] + phi_batch[:, 3]) * 0.5) - ((phi_init_torch[0] + phi_init_torch[3]) * 0.5))**2
            pen7  = ((phi_batch[:, 8] / (phi_batch[:, 2] + eps_div)) - (phi_init_torch[8] / (phi_init_torch[2] + eps_div)))**2
            pen8  = ((phi_batch[:, 11] / (phi_batch[:, 5] + eps_div)) - (phi_init_torch[11] / (phi_init_torch[5] + eps_div)))**2
            pen9  = ((phi_batch[:, 7] / (phi_batch[:, 1] + eps_div)) - (phi_init_torch[7] / (phi_init_torch[1] + eps_div)))**2
            pen10 = ((phi_batch[:, 10] / (phi_batch[:, 4] + eps_div)) - (phi_init_torch[10] / (phi_init_torch[4] + eps_div)))**2

            denom11 = (phi_batch[:, 3] - phi_batch[:, 0]).clone()
            denom11 = torch.where(torch.abs(denom11) < eps_div, torch.full_like(denom11, eps_div), denom11)
            val11 = 2.0 * (phi_batch[:, 6] - phi_batch[:, 0]) / denom11
            init_denom11 = (phi_init_torch[3] - phi_init_torch[0])
            if abs(init_denom11) < eps_div:
                init_denom11 = eps_div
            init_val11 = 2.0 * (phi_init_torch[6] - phi_init_torch[0]) / init_denom11
            pen11 = (val11 - init_val11)**2

            denom12 = (phi_batch[:, 3] - phi_batch[:, 0]).clone()
            denom12 = torch.where(torch.abs(denom12) < eps_div, torch.full_like(denom12, eps_div), denom12)
            val12 = 2.0 * (phi_batch[:, 3] - phi_batch[:, 9]) / denom12
            init_denom12 = (phi_init_torch[3] - phi_init_torch[0])
            if abs(init_denom12) < eps_div:
                init_denom12 = eps_div
            init_val12 = 2.0 * (phi_init_torch[3] - phi_init_torch[9]) / init_denom12
            pen12 = (val12 - init_val12)**2

            penalty = (pen1 + pen2 + pen3 + pen4 + pen5 + pen6 +
                       pen7 + pen8 + pen9 + pen10 + pen11 + pen12).sum()

            loss = loss + reg_lambda * penalty

        loss.backward()
        return loss

    optimizer.step(closure)

    # --- extract fitted raw params (same as original) ---
    params = model.raw.detach().cpu().numpy()  # (B,13)

    # --- compute residuals & ssr ---
    with torch.no_grad():
        y_pred = model(x_t).cpu().numpy()           # (B,L)
        resid = y_norm.cpu().numpy() - y_pred       # (B,L)
        ssr = np.sum(resid**2, axis=1)              # (B,)

    # compute N_eff
    total_length = L * pixel_size
    N_eff = total_length / corr_length

    # --- covariance: per-profile JTJ + R (GN) and df_eff ---
    covs = np.zeros((B, 13, 13), dtype=float)
    dof_array = np.zeros(B, dtype=float)
    for b in range(B):
        raw_b = params[b]  # raw
        sig_b = 1.0 / (1.0 + np.exp(-raw_b))
        theta_b = lb_abs + (ub_abs - lb_abs) * sig_b  # (13,)

        phi_b = theta_to_phi(theta_b, pixel_size, i0)  # (13,) numpy
        
        # Jacobian of model outputs wrt phi (L,13)
        old_params_b = phi_b.tolist()
        Jb = gaussian_quad_jac(np.arange(L), *old_params_b)  # (L,13)
        JTJ = Jb.T @ Jb

        # build R (Gauss-Newton)
        R = build_R_GN(phi_b, reg_lambda)

        H = JTJ + R
        try:
            invH = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            invH = np.linalg.pinv(H)

        # effective number of parameters
        df_eff = np.trace(invH @ JTJ)
        #print(df_eff)
        dof_b = N_eff - df_eff
        if dof_b <= 0:
            dof_b = max(1.0, N_eff * 0.01)
            print('dof < 0 !!!')

        covs[b] = invH * (ssr[b] / dof_b)
        dof_array[b] = float(dof_b)

    return params, covs, ssr, dof_array


class ThicknessCalculator:
    """GPU‐accelerated core thickness computation, now with 4 Gaussians."""
    def __init__(self,
                 volume: np.ndarray,
                 pixel_size: float,
                 mask3d: Optional[np.ndarray] = None):
        self.volume = volume
        self.pixel_size = pixel_size
        self.mask3d = mask3d

        # 2D masks
        self.mask_2d: Optional[np.ndarray] = None
        self.intensity_mask_2d: Optional[np.ndarray] = None
        self.fitting_mask: Optional[np.ndarray] = None

        # computed fields
        self.smoothed_volume: Optional[np.ndarray] = None
        self.width: Optional[np.ndarray] = None
        self.width_sigma: Optional[np.ndarray] = None
        self.loss: Optional[np.ndarray] = None
        self.template_map: Optional[np.ndarray] = None

        # templates: list of (init_abs, lb_abs, ub_abs, (z_lo, z_hi)) in Å
        self.fit_templates: List[Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[float, float]]] = []

        self.theta_map: Optional[np.ndarray] = None   # will hold fitted θ in Å

    def smooth(self, sigma_ang: float = 9.0) -> None:
        """Smooth each Z‐slice with a 2D Gaussian filter."""
        sigma_px = sigma_ang / self.pixel_size
        r = round(3 * sigma_px)
        x = np.linspace(-r, r, 2*r+1)
        X, Y = np.meshgrid(x, x)
        kernel = np.exp(-(X**2 + Y**2) / (2 * sigma_px**2))
        self.smoothed_volume = np.stack([
            fftconvolve(slice_, kernel, mode='same')
            for slice_ in self.volume
        ], axis=0)

    def _choose_mask(self) -> Optional[np.ndarray]:
        if self.fitting_mask is not None:
            return self.fitting_mask
        if self.intensity_mask_2d is not None:
            return self.intensity_mask_2d
        return self.mask_2d

    def compute_thickness(self,
                          tol: float = 1e-3,
                          reg_lambda: float = 0.0,
                          corr_length: float = 6/1.7741,
                          max_rmse: float = 0.1,
                          amp_threshold: float = 0.05) -> None:
        """Batched 4‑Gaussian fitting with the same error logic as before."""
        assert self.smoothed_volume is not None, "Call smooth() first."
        mask = self._choose_mask()
        zdim, ydim, xdim = self.smoothed_volume.shape

        self.width        = np.full((ydim, xdim), np.nan)
        self.width_sigma  = np.full((ydim, xdim), np.nan)
        self.loss         = np.full((ydim, xdim), np.nan)
        self.template_map = np.full((ydim, xdim),   np.nan)
        self.theta_map    = np.full((ydim, xdim, 13), np.nan)

        unassigned = np.ones((ydim, xdim), dtype=bool)
        if mask is not None:
            unassigned &= (mask != 0)

        for t_idx, (init_abs, lb_abs, ub_abs, (z_lo, z_hi)) in enumerate(self.fit_templates):
            if not unassigned.any():
                break

            i0 = round(z_lo / self.pixel_size)
            i1 = round(z_hi / self.pixel_size)
            L  = i1 - i0

            ys, xs = np.nonzero(unassigned)
            B = len(xs)
            if B == 0:
                break

            segments = np.zeros((B, L), dtype=float)
            coords   = []
            for idx, (y, x_) in enumerate(zip(ys, xs)):
                segments[idx] = self.smoothed_volume[i0:i1, y, x_]
                coords.append((y, x_))

            params_b, covs_b, ssr_b, dof_array = fit_batch_profile_gpu(
                segments,
                init_abs, lb_abs, ub_abs,
                i0,
                self.pixel_size,
                corr_length=corr_length,
                reg_lambda=reg_lambda,
            )

            for k, (y, x_) in enumerate(coords):
                p   = params_b[k]
                cov = covs_b[k]
                ssr = ssr_b[k]
                theta = lb_abs + (ub_abs - lb_abs) * torch.sigmoid(torch.from_numpy(p).to('cpu')).detach().numpy()

                # start with per-profile dof
                dof_k = float(dof_array[k])

                tolmask = [0,1,2,3,4,5,6,7,8,9,10,11,12]
                
                if theta[10] < amp_threshold:
                    #dof_k += 3
                    #print('A3 is small')
                    tolmask = [0,1,3,4,5,7,8,9,11,12]
                if theta[11] < amp_threshold:
                    #dof_k += 3
                    #print('A4 is small')
                    tolmask = [0,1,2,4,5,6,8,9,10,12]
                if theta[10] < amp_threshold and theta[11] < amp_threshold:
                    tolmask = [0,1,4,5,8,9,12]

                if (np.isclose(theta[tolmask], lb_abs[tolmask], rtol=tol).any() or
                    np.isclose(theta[tolmask], ub_abs[tolmask], rtol=tol).any()):
                    continue

                rmse = np.sqrt(ssr / dof_k)
                if rmse > max_rmse:
                    continue

                dz = theta[1] * 2
                self.width[y, x_]       = abs(dz)
                var                       = cov[0,0] + cov[3,3] - 2*cov[0,3]
                self.width_sigma[y, x_]   = np.sqrt(max(var, 0)) * self.pixel_size
                self.loss[y, x_]          = rmse
                self.template_map[y, x_]  = t_idx
                self.theta_map[y, x_, :] = theta

                unassigned[y, x_] = False