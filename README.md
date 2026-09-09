Computes nanodisk thickness from .mrc density maps. Nanodisk should be orthogonal to z-axis. Example workflow provided in jupyter notebook

## Membrane Thickness Computation

Density is aligned along the symmetry axis so that the membrane lies parallel to the XY plane. A tight mask is generated around the protein and density inside it is zeroed out to exclude the protein from subsequent analysis. To improve the signal‑to‑noise ratio, the density is smoothed in the XY direction to a 6 Å resolution – corresponding to the characteristic local resolution of the lipid bilayer – by convolution with a Gaussian kernel. Only regions with sufficient density in the membrane are selected for further processing.

For each selected pixel, an intensity profile along the Z‑axis is extracted and fitted with a sum of four Gaussians plus a constant bias (13 parameters total). The fitting is performed using the L‑BFGS optimizer in PyTorch, batched over all pixels for efficiency.

### Parameterisation and bounds

To keep the optimisation within physically sensible limits, the raw parameters are reparameterised using the sigmoid function. Each parameter $\theta_i$ is mapped from an unbounded variable $r_i$ via

$$
\theta_i = lb_i + (ub_i - lb_i) \cdot \sigma(r_i),
$$

where $lb_i$ and $ub_i$ are the user‑defined lower and upper bounds. This ensures that all parameters remain within their allowed ranges during optimisation while the optimizer works with unconstrained variables.

The thirteen parameters, in order, are:

| Index | Symbol      | Meaning                                     |
|-------|-------------|---------------------------------------------|
| 0     | $m_0$        | Centre of the two main Gaussians (Å)        |
| 1     | $d_0$        | Half‑distance between main Gaussians (Å)    |
| 2     | $d_1$        | Fraction of $d_0$ for satellite‑1 offset    |
| 3     | $d_2$        | Fraction of $d_0$ for satellite‑2 offset    |
| 4     | $\sigma_1$   | Width of main Gaussian 1 (Å)                |
| 5     | $\sigma_2$   | Width of main Gaussian 2 (Å)                |
| 6     | $f_{\sigma_3}$ | Fraction of $\sigma_1$ for satellite 3   |
| 7     | $f_{\sigma_4}$ | Fraction of $\sigma_2$ for satellite 4   |
| 8     | $A_1$        | Amplitude of main Gaussian 1                |
| 9     | $A_2$        | Amplitude of main Gaussian 2                |
| 10    | $f_{A_3}$    | Fraction of $A_1$ for satellite 3           |
| 11    | $f_{A_4}$    | Fraction of $A_2$ for satellite 4           |
| 12    | $b$          | Constant bias                               |

From these parameters, the actual positions, widths, and amplitudes of the four Gaussians are derived:

$$
m_1 = m_0 - d_0, \quad m_2 = m_0 + d_0,
$$
$$
m_3 = m_1 + d_1\cdot d_0, \quad m_4 = m_2 - d_2\cdot d_0,
$$
$$
\sigma_3 = \sigma_1 \cdot f_{\sigma_3}, \quad \sigma_4 = \sigma_2 \cdot f_{\sigma_4},
$$
$$
A_3 = A_1 \cdot f_{A_3}, \quad A_4 = A_2 \cdot f_{A_4}.
$$

All positions and widths are converted to pixel units relative to the profile start before evaluation.

### Regularization

The four‑Gaussian model is intentionally over‑parameterised to accommodate a wide range of profile shapes, but this flexibility can cause numerical instability or unphysical fits when satellite peaks are weak or poorly resolved. To mitigate this, we add a quadratic regularization term that encourages the fitted profile to retain certain relationships observed in the best‑resolved regions. These relationships are expressed as twelve scalar functions $f_k(\boldsymbol{\phi})$ of the *pixel‑relative* parameters

$$
\boldsymbol{\phi} = [m_1, \sigma_1, A_1, m_2, \sigma_2, A_2, m_3, \sigma_3, A_3, m_4, \sigma_4, A_4, b]^\top.
$$

Each function is zero when the corresponding relationship holds exactly at the initial guess $\boldsymbol{\phi}^{(0)}$ (provided by the template). The total regularization penalty is

$$
\mathcal{R}(\boldsymbol{\phi}) = \sum_{k=1}^{12} f_k(\boldsymbol{\phi})^2,
$$

and the complete loss function becomes

$$
\mathcal{L} = \underbrace{\sum_{\text{pixels}} \left( y_{\text{data}} - y_{\text{model}}(\boldsymbol{\phi}) \right)^2}_{\text{SSR}} \;+\; \lambda \,\mathcal{R}(\boldsymbol{\phi}),
$$

where $\lambda$ is the user‑specified `reg_lambda`.

The twelve constraints are summarised below:

| # | Constraint $f_k(\boldsymbol{\phi})$ | Purpose |
|---|--------------------------------------|---------|
| 1 | $b - b^{(0)}$ | Keep bias near initial value |
| 2 | $A_1 + b - (A_1^{(0)} + b^{(0)})$ | Stabilise sum of first amplitude and bias |
| 3 | $A_2 + b - (A_2^{(0)} + b^{(0)})$ | Stabilise sum of second amplitude and bias |
| 4 | $A_1 - A_2$ | Encourage equal main amplitudes |
| 5 | $\sigma_1 - \sigma_2$ | Encourage equal main widths |
| 6 | $\frac{m_1 + m_2}{2} - \frac{m_1^{(0)} + m_2^{(0)}}{2}$ | Hold centre of main peaks near initial value |
| 7 | $\frac{A_3}{A_1} - \frac{A_3^{(0)}}{A_1^{(0)}}$ | Preserve amplitude ratio of satellite 3 |
| 8 | $\frac{A_4}{A_2} - \frac{A_4^{(0)}}{A_2^{(0)}}$ | Preserve amplitude ratio of satellite 4 |
| 9 | $\frac{\sigma_3}{\sigma_1} - \frac{\sigma_3^{(0)}}{\sigma_1^{(0)}}$ | Preserve width ratio of satellite 3 |
| 10 | $\frac{\sigma_4}{\sigma_2} - \frac{\sigma_4^{(0)}}{\sigma_2^{(0)}}$ | Preserve width ratio of satellite 4 |
| 11 | $\frac{2(m_3 - m_1)}{m_2 - m_1} - \frac{2(m_3^{(0)} - m_1^{(0)})}{m_2^{(0)} - m_1^{(0)}}$ | Keep relative position of satellite 3 |
| 12 | $\frac{2(m_2 - m_4)}{m_2 - m_1} - \frac{2(m_2^{(0)} - m_4^{(0)})}{m_2^{(0)} - m_1^{(0)}}$ | Keep relative position of satellite 4 |

The penalty functions are designed to be **scale‑invariant** where possible (ratios, normalised positions), making the regularization equally meaningful for profiles with different overall intensities or thicknesses.

During optimisation, the constraints are evaluated in PyTorch using the current $\boldsymbol{\phi}$ and the initial $\boldsymbol{\phi}^{(0)}$. For the error estimation (below), the Gauss‑Newton approximation of the regularisation Hessian is built analytically: each term contributes $2\,\lambda\,\nabla f_k \nabla f_k^\top$ to the matrix $R$.

### Template fitting

To achieve good fit quality across all regions of the nanodisc, a dictionary of templates is manually defined. Each template specifies:

- a Z‑range for profile extraction,
- initial parameter guesses,
- lower and upper bounds for each parameter.

Templates are tried sequentially for each pixel. The first template that produces a fit satisfying all acceptance criteria (e.g., parameters not stuck at bounds, RMSE below a threshold) is accepted and its results are stored. This approach accommodates local variations in the membrane profile shape.

### Error estimation

Parameter uncertainties are estimated using the Hessian matrix computed in the Gauss‑Newton approximation from **analytical Jacobians** of the model with respect to the $\boldsymbol{\phi}$ parameters, plus the regularisation matrix $R$:

$$
H = J^\top J + R,
$$

where $J$ is the Jacobian of the residuals and $R$ is the regularisation contribution described above.

The covariance matrix of the fitted parameters is then

$$
\Sigma = H^{-1} \cdot \frac{\text{SSR}}{\text{dof}},
$$

with $\text{SSR}$ being the sum of squared residuals and $\text{dof}$ the effective degrees of freedom.

The **effective number of independent data points** is calculated as

$$
N_{\text{eff}} = \frac{L \cdot \text{pixel\_size}}{\text{corr\_length}},
$$

where $L$ is the number of pixels in the profile and `corr_length` is the correlation length corresponding to the 6 Å smoothing (here $6 / 1.7741$ Å). The effective number of parameters is

$$
p_{\text{eff}} = \text{trace}\!\left( H^{-1} J^\top J \right),
$$

and the degrees of freedom are

$$
\text{dof} = N_{\text{eff}} - p_{\text{eff}}.
$$

If `dof` becomes non‑positive, it is clamped to a small positive value to avoid division by zero.

This framework provides a robust, quantitative estimate of the membrane thickness and its uncertainty for every pixel in the map.