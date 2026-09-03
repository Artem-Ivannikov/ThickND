import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from matplotlib.colors import ListedColormap
from IPython.display import display, clear_output
from typing import Optional, Tuple
import torch
from calc import gaussian_quad, theta_to_phi

from calc import ThicknessCalculator, fit_batch_profile_gpu

def plot_2d_image(data: np.ndarray,
                  pixel_size: float,
                  title: str,
                  cbar_label: str,
                  cmap: str = 'viridis') -> None:
    """Plot a 2D image with axes in Ångströms and a colorbar."""
    ydim, xdim = data.shape
    extent = [0, xdim * pixel_size, 0, ydim * pixel_size]
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = plt.cm.get_cmap(cmap).copy()
    cm.set_bad(color='gray')
    im = ax.imshow(data, cmap=cm, extent=extent, origin='lower', aspect='auto')
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_title(title)
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.format_coord = lambda x, y: f"x={x:.1f}Å, y={y:.1f}Å"
    plt.show()


class ThicknessViewer:
    """Visualization and interactive tools for ThicknessCalculator."""
    def __init__(self, calculator: ThicknessCalculator):
        self.calc = calculator

    def plot_results(self,
                     field: str = 'width',
                     cmap: str = 'viridis') -> None:
        """
        Plot a computed 2D field (e.g., 'width', 'loss')
        """
        data = getattr(self.calc, field, None)
        assert data is not None, f"Field '{field}' not found."
        # choose mask
        mask = None
        if getattr(self.calc, 'fitting_mask', None) is not None:
            mask = self.calc.fitting_mask.astype(bool)
        elif self.calc.intensity_mask_2d is not None:
            mask = self.calc.intensity_mask_2d.astype(bool)
        elif self.calc.mask_2d is not None:
            mask = self.calc.mask_2d.astype(bool)
        pix = self.calc.pixel_size
        ydim, xdim = data.shape
        extent = [0, xdim * pix, 0, ydim * pix]

        # define regions
        valid = mask & ~np.isnan(data) if mask is not None else ~np.isnan(data)
        #fail = mask & np.isnan(data) if mask is not None else np.zeros_like(data, bool)
        outside = (~mask) if mask is not None else np.zeros_like(data, bool)

        fig, ax = plt.subplots(figsize=(8, 6))
        # background
        ax.set_facecolor('red')
        # outside-mask dark gray
        if outside.any():
            out_img = np.zeros((ydim, xdim, 4))
            out_img[outside] = (0.5, 0.5, 0.5, 1.0)
            ax.imshow(out_img, extent=extent, origin='lower', interpolation='nearest', zorder=1)
        # data
        cm_data = plt.cm.get_cmap(cmap).copy()
        data_img = np.ma.masked_where(~valid, data)
        im = ax.imshow(data_img, cmap=cm_data,
                       extent=extent, origin='lower', interpolation='nearest', zorder=2)
        fig.colorbar(im, ax=ax, label=field)
        # failures red
        #if fail.any():
        #    fail_img = np.zeros((ydim, xdim, 4))
        #    fail_img[fail] = (1.0, 0.0, 0.0, 1.0)
        #    ax.imshow(fail_img, extent=extent, origin='lower', interpolation='nearest', zorder=3)
        ax.set_title(f"{field.capitalize()} Map")
        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")
        ax.format_coord = lambda x, y: f"x={x:.1f}Å, y={y:.1f}Å"
        plt.show()

    def plot_template_map(self) -> None:
        """Plot template indices per pixel with axes in Å, highlighting failures in gray."""
        tmap = self.calc.template_map
        assert tmap is not None, "Run compute_thickness first."
        pix = self.calc.pixel_size
        ydim, xdim = tmap.shape
        extent = [0, xdim * pix, 0, ydim * pix]

        # determine masks
        mask = None
        if getattr(self.calc, 'fitting_mask', None) is not None:
            mask = self.calc.fitting_mask.astype(bool)
        elif self.calc.intensity_mask_2d is not None:
            mask = self.calc.intensity_mask_2d.astype(bool)
        elif self.calc.mask_2d is not None:
            mask = self.calc.mask_2d.astype(bool)

        valid = mask & (tmap >= 0) if mask is not None else (tmap >= 0)
        #fail = mask & (tmap < 0) if mask is not None else np.zeros_like(tmap, bool)
        #outside = mask is not None and ~mask or np.zeros_like(tmap, bool)
        outside = (~mask) if mask is not None else np.zeros_like(tmap, bool)

        fig, ax = plt.subplots(figsize=(8, 6))
        # background
        ax.set_facecolor('lightgray')
        # outside-mask dark gray
        if outside.any():
            out_img = np.zeros((ydim, xdim, 4))
            out_img[outside] = (0.33, 0.33, 0.33, 1.0)
            ax.imshow(out_img, extent=extent, origin='lower', interpolation='nearest', zorder=1)
        # data colormap
        cmap = plt.get_cmap('tab10', len(self.calc.fit_templates))
        data_img = np.ma.masked_where(~valid, tmap)
        im = ax.imshow(data_img, origin='lower', extent=extent,
                       interpolation='nearest', cmap=cmap, zorder=2)
        fig.colorbar(im, ax=ax, ticks=range(len(self.calc.fit_templates)), label='Template Index')
        # failures in red
        #if fail.any():
        #    fail_img = np.zeros((ydim, xdim, 4))
        #    fail_img[fail] = (1.0, 0.0, 0.0, 1.0)
        #    ax.imshow(fail_img, extent=extent, origin='lower', interpolation='nearest', zorder=3)

        ax.set_title('Template Map')
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.format_coord = lambda x, y: f"x={x:.1f}Å, y={y:.1f}Å, tpl={int(tmap[int(y/pix), int(x/pix)]) if 0<=int(y/pix)<ydim and 0<=int(x/pix)<xdim else 'NA'}"
        plt.show()

    def plot_projection(self,
                        method: str = 'mean',
                        smoothed: bool = False) -> None:
        """Plot a 2D projection of the volume."""
        vol = self.calc.smoothed_volume if smoothed else self.calc.volume
        proj = getattr(vol, method)(axis=0)
        title = f"2D Projection ({method}, {'smoothed' if smoothed else 'raw'})"
        plot_2d_image(proj,
                      self.calc.pixel_size,
                      title,
                      "Density",
                      cmap='gray')

    def interactive_slice_viewer(self,
                                 smoothed: bool = False,
                                 init_z: float = 0.0) -> None:
        """Interactive Z-slice viewer with slider in Å."""
        vol = self.calc.smoothed_volume if smoothed else self.calc.volume
        assert vol is not None, "Volume not loaded."
        pix = self.calc.pixel_size
        zdim = vol.shape[0]
        max_z = (zdim - 1) * pix
        init = np.clip(init_z, 0, max_z)

        slider = widgets.FloatSlider(
            value=init, min=0, max=max_z, step=pix,
            description='Z (Å):', continuous_update=False,
            layout=widgets.Layout(width='90%')
        )
        out = widgets.Output()

        def _update(change=None):
            z0 = slider.value
            idx = int(z0 / pix)
            with out:
                clear_output(wait=True)
                plot_2d_image(vol[idx], pix,
                              f"Slice Z={z0:.1f} Å", "Density",
                              cmap='gray')

        slider.observe(_update, names='value')
        display(widgets.VBox([slider, out]))
        _update()

    def interactive_intensity_mask(self,
                                   init_z_range: Optional[Tuple[float, float]] = None,
                                   init_thresh: float = 0.5,
                                   plot: bool = True) -> None:
        """Interactive creation of 2D intensity mask from smoothed volume."""
        vol = self.calc.smoothed_volume
        assert vol is not None, "Run smooth() first."
        pix = self.calc.pixel_size
        zdim = vol.shape[0]
        max_z = (zdim - 1) * pix
        z0, z1 = init_z_range or (0.0, max_z)

        zslider = widgets.FloatRangeSlider(
            value=(z0, z1), min=0, max=max_z, step=pix,
            description='Z bounds (Å):', continuous_update=False,
            layout=widgets.Layout(width='600px')
        )
        tslider = widgets.FloatSlider(
            value=init_thresh, min=0.0, max=1.0, step=0.01,
            description='Threshold:', continuous_update=False,
            layout=widgets.Layout(width='400px')
        )
        btn = widgets.Button(description="Save Intensity Mask", button_style='success')
        out = widgets.Output()

        def _compute_mask(zmin, zmax, thresh):
            i0, i1 = int(zmin / pix), int(zmax / pix) + 1
            comp = vol[i0:i1].mean(axis=0)
            norm = (comp - comp.min()) / (np.ptp(comp) + 1e-8)
            return (norm > thresh).astype(int)

        def _update(*_):
            zmin, zmax = zslider.value
            mask2d = _compute_mask(zmin, zmax, tslider.value)
            if plot:
                with out:
                    clear_output(wait=True)
                    plot_2d_image(mask2d, pix,
                                  f"Intensity Mask Z={zmin:.1f}-{zmax:.1f} Å", "Mask",
                                  cmap='gray')

        def _save(_):
            zmin, zmax = zslider.value
            self.calc.intensity_mask_2d = _compute_mask(zmin, zmax, tslider.value)
            with out:
                print("✅ Intensity mask saved to calculator.intensity_mask_2d")

        zslider.observe(_update, names='value')
        tslider.observe(_update, names='value')
        btn.on_click(_save)

        display(widgets.VBox([zslider, tslider, btn, out]))

        # 🔹 Save immediately with initial params
        self.calc.intensity_mask_2d = _compute_mask(z0, z1, init_thresh)
        if not plot:
            return
        _update()


    def interactive_mask_collapse(self,
                                  init_z_range: Optional[Tuple[float, float]] = None,
                                  init_thresh: float = 0.99,
                                  plot: bool = True) -> None:
        """Interactive projection of 3D mask to 2D via thresholding."""
        mask3d = self.calc.mask3d
        assert mask3d is not None, "Load a 3D mask first."
        pix = self.calc.pixel_size
        zdim = mask3d.shape[0]
        max_z = (zdim - 1) * pix
        z0, z1 = init_z_range or (0.0, max_z)

        zslider = widgets.FloatRangeSlider(
            value=(z0, z1), min=0, max=max_z, step=pix,
            description='Z bounds (Å):', continuous_update=False,
            layout=widgets.Layout(width='90%')
        )
        tslider = widgets.FloatSlider(
            value=init_thresh, min=0.0, max=1.0, step=0.01,
            description='Threshold:', continuous_update=False,
            layout=widgets.Layout(width='90%')
        )
        btn = widgets.Button(description="Save Mask", button_style='success')
        out = widgets.Output()

        def _compute_mask(zmin, zmax, thresh):
            i0, i1 = int(zmin / pix), int(zmax / pix) + 1
            collapsed = ~np.any(mask3d[i0:i1] < thresh, axis=0)
            cy, cx = np.array(collapsed.shape) // 2
            hy, hx = np.array(self.calc.volume.shape[1:]) // 2
            return collapsed[cy-hy:cy-hy+self.calc.volume.shape[1],
                             cx-hx:cx-hx+self.calc.volume.shape[2]]

        def _update(*_):
            zmin, zmax = zslider.value
            mask2d = _compute_mask(zmin, zmax, tslider.value)
            self.calc.mask_2d = mask2d.astype(int)
            if plot:
                with out:
                    clear_output(wait=True)
                    plot_2d_image(mask2d, pix,
                                  f"Mask 2D Z={zmin:.1f}-{zmax:.1f} Å", "Mask",
                                  cmap='gray')

        def _save(_):
            zmin, zmax = zslider.value
            self.calc.mask_2d = _compute_mask(zmin, zmax, tslider.value).astype(int)
            with out:
                print("✅ Mask saved to calculator.mask_2d")

        zslider.observe(_update, names='value')
        tslider.observe(_update, names='value')
        btn.on_click(_save)

        display(widgets.VBox([zslider, tslider, btn, out]))

        # 🔹 Save immediately with initial params
        self.calc.mask_2d = _compute_mask(z0, z1, init_thresh).astype(int)
        if not plot:
            return
        _update()


    def create_combined_mask(self) -> None:
        """Combine projection and intensity masks into fitting_mask."""
        m2d = self.calc.mask_2d
        im2d = self.calc.intensity_mask_2d
        assert m2d is not None and im2d is not None, "Create both masks first." 
        if m2d.shape != im2d.shape:
            raise ValueError("Mask shapes mismatch.")
        self.calc.fitting_mask = (m2d & im2d).astype(int)
        plot_2d_image(self.calc.fitting_mask,
                      self.calc.pixel_size,
                      "Combined Fitting Mask",
                      "Mask",
                      cmap='gray')
    
    def plot_profile_point(self,
                        x_angs: float,
                        y_angs: float,
                        template_idx: Optional[int] = None,
                        full_range: bool = False,
                        corr_length: float = 6/1.7741,
                        reg_lambda: float = 0.0,
                        max_rmse: float = 0.1,
                        tol: float = 1e-3) -> None:
        """
        Plot intensity profile at (x,y) in Å using the batched GPU fitter,
        under the 4-Gaussian reparam scheme.

        Parameters
        ----------
        x_angs, y_angs : float
            Coordinates in Å (will be rounded to nearest pixel).
        template_idx : int or None
            If given, only this template is tried; otherwise all templates are attempted.
        full_range : bool
            If True, plot the full Z range of the profile; otherwise only the fitted segment.
        corr_length, reg_lambda : passed to the fitter.
        max_rmse, tol : acceptance criteria (same meaning as in compute_thickness).
        """
        calc = self.calc
        pix  = calc.pixel_size

        # 1) pixel coordinates, raw profile, full-Z axis in Å
        x_pix, y_pix = round(x_angs/pix), round(y_angs/pix)
        prof   = calc.smoothed_volume[:, y_pix, x_pix].astype(float)
        zdim   = prof.size
        z_full = np.arange(zdim) * pix

        # 2) choose templates
        tlist = ([template_idx]
                if template_idx is not None
                else list(range(len(calc.fit_templates))))

        for t_idx in tlist:
            init_abs, lb_abs, ub_abs, (z_lo, z_hi) = calc.fit_templates[t_idx]
            i0 = round(z_lo/pix)
            i1 = round(z_hi/pix)
            seg = prof[i0:i1]
            if seg.size == 0:
                continue

            # 3) one-element batch fit
            params_b, covs_b, ssr_b, dof_array = fit_batch_profile_gpu(
                seg[np.newaxis,:],
                init_abs, lb_abs, ub_abs,
                i0, pix, corr_length, reg_lambda
            )

            p   = params_b[0]          # (13,) raw logits
            cov = covs_b[0]
            ssr = ssr_b[0]
            dof = dof_array[0]

            # Convert raw logits to physical theta (absolute Å)
            theta = lb_abs + (ub_abs - lb_abs) * torch.sigmoid(torch.from_numpy(p).to('cpu')).detach().numpy()

            # Acceptance check (same as compute_thickness)
            # We skip the template if any of the key parameters hit bounds or RMSE too high
            tolmask = list(range(13))
            # (You can replicate the logic from compute_thickness if needed; here we use a simple bound check)
            if (np.isclose(theta[tolmask], lb_abs[tolmask], rtol=tol).any() or
                np.isclose(theta[tolmask], ub_abs[tolmask], rtol=tol).any()):
                continue

            rmse = np.sqrt(ssr / dof) if dof > 0 else np.inf
            if rmse > max_rmse:
                continue

            # Good fit found – now plot
            # Convert theta to phi (pixel‑relative Gaussian parameters)
            phi = theta_to_phi(theta, pix, i0)   # numpy array (13,)
            m1_px, s1_px, A1, m2_px, s2_px, A2, m3_px, s3_px, A3, m4_px, s4_px, A4, bias = phi

            # Compute absolute positions in Å for display
            m1 = (m1_px + i0) * pix
            m2 = (m2_px + i0) * pix
            m3 = (m3_px + i0) * pix
            m4 = (m4_px + i0) * pix
            sigma1 = s1_px * pix
            sigma2 = s2_px * pix
            sigma3 = s3_px * pix
            sigma4 = s4_px * pix
            thickness = m2 - m1

            print(f"Fit at ({x_angs:.1f}Å, {y_angs:.1f}Å), tpl={t_idx}:")
            print(f"  Peak1: z={m1:.2f}Å, σ={sigma1:.2f}Å, A={A1:.2f}")
            print(f"  Peak2: z={m2:.2f}Å, σ={sigma2:.2f}Å, A={A2:.2f}")
            print(f"  Peak3: z={m3:.2f}Å, σ={sigma3:.2f}Å, A={A3:.2f}")
            print(f"  Peak4: z={m4:.2f}Å, σ={sigma4:.2f}Å, A={A4:.2f}")
            print(f"  Offset={bias:.2f}; Thickness={thickness:.2f}Å")

            # Plot
            plt.figure(figsize=(6,4))
            if full_range:
                # Normalise both data and model by the full profile maximum
                norm_factor = prof.max() if prof.max() > 0 else 1.0
                norm_prof = prof / norm_factor
                plt.plot(z_full, norm_prof, 'o', alpha=0.6, label='Raw Data')
                z_plot = z_full
                x_plot = z_plot / pix - i0
                # Scale amplitudes to match full-profile normalisation
                scale = seg.max() / norm_factor if norm_factor != 0 else 1.0
                fit_curve = gaussian_quad(x_plot,
                                        m1_px, s1_px, A1 * scale,
                                        m2_px, s2_px, A2 * scale,
                                        m3_px, s3_px, A3 * scale,
                                        m4_px, s4_px, A4 * scale,
                                        bias * scale)
            else:
                z_seg = np.linspace(i0*pix, i1*pix, seg.size)
                norm_seg = seg / (seg.max() + 1e-8)
                plt.plot(z_seg, norm_seg, 'o', alpha=0.8, label='Segment')
                z_plot = z_seg
                x_plot = z_plot / pix - i0
                # Amplitudes already normalised to segment max
                fit_curve = gaussian_quad(x_plot,
                                        m1_px, s1_px, A1,
                                        m2_px, s2_px, A2,
                                        m3_px, s3_px, A3,
                                        m4_px, s4_px, A4,
                                        bias)
            plt.plot(z_plot, fit_curve, '-', lw=2, label='Fit')
            plt.xlabel('Z (Å)')
            plt.ylabel('Normalized Intensity')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            return

        print(f"No template succeeded at ({x_angs:.1f}Å, {y_angs:.1f}Å).")
        
    def plot_profile_from_map(
            self,
            x_angs: float,
            y_angs: float,
            full_range: bool = False,
            ax=None,
            *,
            verbose: bool = True,
            annotate_thickness: bool = False,
            thickness_label_fontsize: int = 9,
            thickness_label_bbox: Optional[dict] = None,
        ):
        """
        Plot the profile using precomputed θ from compute_thickness, without refitting.

        Parameters
        ----------
        x_angs, y_angs : float
            Coordinates in Å.
        full_range : bool
            If True, plot the full Z range; otherwise only the fitted segment.
        ax : matplotlib.axes.Axes or None
            If given, draw on this axis; otherwise create a new figure.
        verbose : bool
            Print fit parameters.
        annotate_thickness : bool
            Add a text box with thickness.
        thickness_label_fontsize : int
            Font size for the annotation.
        thickness_label_bbox : dict or None
            Bbox properties for the annotation.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
            created_fig = True

        calc = self.calc
        pix = calc.pixel_size

        # Coordinates
        i = round(y_angs / pix)
        j = round(x_angs / pix)

        # Check that a fit exists
        t_idx = calc.template_map[i, j]
        if t_idx < 0:
            if verbose:
                print(f"No fit stored at ({x_angs:.1f}Å,{y_angs:.1f}Å)")
            return ax

        theta = calc.theta_map[i, j, :]
        if np.isnan(theta).any():
            if verbose:
                print(f"No fit stored at ({x_angs:.1f}Å,{y_angs:.1f}Å)")
            return ax

        # Retrieve template info for segment boundaries
        _, _, _, (z_lo, z_hi) = calc.fit_templates[t_idx]
        i0 = round(z_lo / pix)
        i1 = round(z_hi / pix)

        prof = calc.smoothed_volume[:, i, j].astype(float)
        zdim = prof.size
        z_full = np.arange(zdim) * pix
        seg = prof[i0:i1]

        # Convert stored theta (absolute) to phi (pixel‑relative)
        phi = theta_to_phi(theta, pix, i0)   # numpy array (13,)
        m1_px, s1_px, A1, m2_px, s2_px, A2, m3_px, s3_px, A3, m4_px, s4_px, A4, bias = phi

        # Compute absolute positions for display
        m1 = (m1_px + i0) * pix
        m2 = (m2_px + i0) * pix
        m3 = (m3_px + i0) * pix
        m4 = (m4_px + i0) * pix
        sigma1 = s1_px * pix
        sigma2 = s2_px * pix
        sigma3 = s3_px * pix
        sigma4 = s4_px * pix
        thickness = m2 - m1

        if verbose:
            print(theta)
            print(f"Fit at ({x_angs:.1f}Å, {y_angs:.1f}Å), tpl={t_idx}:")
            print(f"  Peak1: z={m1:.2f}Å, σ={sigma1:.2f}Å, A={A1:.2f}")
            print(f"  Peak2: z={m2:.2f}Å, σ={sigma2:.2f}Å, A={A2:.2f}")
            print(f"  Peak3: z={m3:.2f}Å, σ={sigma3:.2f}Å, A={A3:.2f}")
            print(f"  Peak4: z={m4:.2f}Å, σ={sigma4:.2f}Å, A={A4:.2f}")
            print(f"  Offset={bias:.2f}; Thickness={thickness:.2f}Å")

        # Plot data and model
        if full_range:
            norm_factor = prof.max() if prof.max() > 0 else 1.0
            norm_prof = prof / norm_factor
            ax.plot(z_full, norm_prof, 'o', alpha=0.6, label='Raw Data')
            z_plot = z_full
            x_plot = z_plot / pix - i0
            # Scale amplitudes to full-profile normalisation
            scale = seg.max() / norm_factor if norm_factor != 0 else 1.0
            fit_curve = gaussian_quad(x_plot,
                                    m1_px, s1_px, A1 * scale,
                                    m2_px, s2_px, A2 * scale,
                                    m3_px, s3_px, A3 * scale,
                                    m4_px, s4_px, A4 * scale,
                                    bias * scale)
        else:
            z_seg = np.linspace(i0 * pix, i1 * pix, seg.size)
            norm_seg = seg / (seg.max() + 1e-8)
            ax.plot(z_seg, norm_seg, 'o', alpha=0.8, label='Segment')
            z_plot = z_seg
            x_plot = z_plot / pix - i0
            fit_curve = gaussian_quad(x_plot,
                                    m1_px, s1_px, A1,
                                    m2_px, s2_px, A2,
                                    m3_px, s3_px, A3,
                                    m4_px, s4_px, A4,
                                    bias)

        ax.plot(z_plot, fit_curve, '-', lw=2, label='Stored Fit')

        if annotate_thickness:
            bbox = thickness_label_bbox if thickness_label_bbox is not None else dict(
                boxstyle='round', fc='white', ec='0.7', alpha=0.9
            )
            ax.text(
                0.5, 0.05,
                f"Thickness = {thickness:.1f} Å",
                transform=ax.transAxes,
                ha='center', va='bottom',
                fontsize=thickness_label_fontsize,
                bbox=bbox,
                clip_on=False
            )

        ax.set_xlabel("Z (Å)")
        ax.set_ylabel("Normalized Intensity")
        ax.set_title(f"Fit at ({x_angs:.1f}Å, {y_angs:.1f}Å)")
        ax.legend()
        ax.grid(True)

        if created_fig:
            fig.tight_layout()
            plt.show()

        return ax
                