import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interactive_output
from IPython.display import display, clear_output
from typing import Optional, Tuple, Callable, Any
import torch
from calc import gaussian_quad, theta_to_phi
from calc import ThicknessCalculator, fit_batch_profile_gpu


class ThicknessViewer:
    """Visualization and interactive tools for ThicknessCalculator."""

    def __init__(self, calculator: ThicknessCalculator):
        self.calc = calculator

    # ---------- helper methods ----------
    def _get_2d_mask(self) -> Optional[np.ndarray]:
        """Return the most appropriate 2D mask or None."""
        for attr in ['fitting_mask', 'intensity_mask_2d', 'mask_2d']:
            mask = getattr(self.calc, attr, None)
            if mask is not None:
                return mask.astype(bool)
        return None

    def _set_angstrom_coord(self, ax, extra_info: Optional[Callable[[int, int], str]] = None) -> None:
        """Set format_coord to display Å, optionally with extra info."""
        pix = self.calc.pixel_size

        def format_coord(x: float, y: float) -> str:
            s = f"x={x:.1f}Å, y={y:.1f}Å"
            if extra_info:
                i, j = int(y / pix), int(x / pix)
                if 0 <= i < self.calc.width.shape[0] and 0 <= j < self.calc.width.shape[1]:
                    s += extra_info(i, j)
            return s

        ax.format_coord = format_coord

    def _plot_masked_map(self,
                         data: np.ndarray,
                         mask: Optional[np.ndarray],
                         cmap: str,
                         title: str,
                         cbar_label: str,
                         cbar_ticks: Optional[list] = None,
                         extra_coord_info: Optional[Callable[[int, int], str]] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Common routine for plotting 2D data with optional mask background.

        If mask is None, all non-NaN pixels are treated as valid.
        """
        pix = self.calc.pixel_size
        ydim, xdim = data.shape
        extent = [0, xdim * pix, 0, ydim * pix]

        if mask is not None:
            valid = mask & ~np.isnan(data)
            outside = ~mask
        else:
            valid = ~np.isnan(data)
            outside = np.zeros_like(data, dtype=bool)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_facecolor('red')  # invalid pixels inside mask (or NaN) shown red

        if outside.any():
            out_img = np.zeros((ydim, xdim, 4))
            out_img[outside] = (0.33, 0.33, 0.33, 1.0)
            ax.imshow(out_img, extent=extent, origin='lower',
                      interpolation='nearest', zorder=1)

        data_img = np.ma.masked_where(~valid, data)
        im = ax.imshow(data_img, cmap=cmap, extent=extent, origin='lower',
                       interpolation='nearest', zorder=2)

        cbar = fig.colorbar(im, ax=ax, label=cbar_label)
        if cbar_ticks is not None:
            cbar.set_ticks(cbar_ticks)

        ax.set_title(title)
        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")
        self._set_angstrom_coord(ax, extra_coord_info)
        return fig, ax

    def _plot_gaussian_profile(self,
                               ax: plt.Axes,
                               prof: np.ndarray,
                               i0: int,
                               i1: int,
                               pix: float,
                               phi: np.ndarray,
                               full_range: bool,
                               label: str = 'Fit') -> plt.Axes:
        """Plot raw/segmented profile and the Gaussian fit on ax."""
        zdim = prof.size
        z_full = np.arange(zdim) * pix
        seg = prof[i0:i1]

        if full_range:
            norm_factor = prof.max() if prof.max() > 0 else 1.0
            norm_prof = prof / norm_factor
            ax.plot(z_full, norm_prof, 'o', alpha=0.6, label='Raw Data')
            z_plot = z_full
            x_plot = z_plot / pix - i0
            scale = seg.max() / norm_factor if norm_factor != 0 else 1.0
            fit_curve = gaussian_quad(x_plot, *phi[:12], phi[12] * scale)
        else:
            z_seg = np.linspace(i0 * pix, i1 * pix, seg.size)
            norm_seg = seg / (seg.max() + 1e-8)
            ax.plot(z_seg, norm_seg, 'o', alpha=0.8, label='Segment')
            z_plot = z_seg
            x_plot = z_plot / pix - i0
            fit_curve = gaussian_quad(x_plot, *phi)

        ax.plot(z_plot, fit_curve, '-', lw=2, label=label)
        return ax

    def _interactive_mask_creator(self,
                                compute_func: Callable[[float, float, float], np.ndarray],
                                save_attr: str,
                                description: str,
                                init_z_range: Optional[Tuple[float, float]] = None,
                                init_thresh: float = 0.5,
                                plot: bool = True) -> None:
        """Generic interactive mask creation with sliders and save button."""
        pix = self.calc.pixel_size
        zdim = self.calc.smoothed_volume.shape[0]
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

        def _compute_current_mask():
            zmin, zmax = zslider.value
            thresh = tslider.value
            return compute_func(zmin, zmax, thresh)

        def _update_preview(*args):
            mask = _compute_current_mask()
            if plot:
                with out:
                    clear_output(wait=True)
                    self._plot_masked_map(mask, None, cmap='gray',
                                        title=description, cbar_label='Mask')
                    plt.show()

        def _save(_):
            mask = _compute_current_mask()
            setattr(self.calc, save_attr, mask.astype(int))
            with out:
                print(f"✅ Mask saved to calculator.{save_attr}")

        # Attach callbacks
        zslider.observe(_update_preview, names='value')
        tslider.observe(_update_preview, names='value')
        btn.on_click(_save)

        # Initial setup: save with initial parameters, then display preview
        initial_mask = compute_func(z0, z1, init_thresh)
        setattr(self.calc, save_attr, initial_mask.astype(int))
        if plot:
            with out:
                clear_output(wait=True)
                self._plot_masked_map(initial_mask, None, cmap='gray',
                                    title=description, cbar_label='Mask')
                plt.show()

        display(widgets.VBox([zslider, tslider, btn, out]))

    # ---------- unified map plotting ----------
    def plot_map(self, field: str = 'width', cmap: str = 'viridis') -> None:
        """
        Plot a computed 2D field (e.g., 'width', 'loss', 'template_map').

        For 'template_map', a categorical colormap is used and the colorbar
        shows integer template indices. Other fields use a continuous colormap.
        """
        data = getattr(self.calc, field, None)
        assert data is not None, f"Field '{field}' not found."

        mask = self._get_2d_mask()

        if field == 'template_map':
            cmap = plt.get_cmap('tab10', len(self.calc.fit_templates))
            cbar_ticks = range(len(self.calc.fit_templates))
            title = 'Template Map'
            cbar_label = 'Template Index'
            extra_info = lambda i, j: f", tpl={'NA' if np.isnan(data[i, j]) else int(data[i, j])}"
        else:
            cbar_ticks = None
            title = f"{field.capitalize()} Map"
            cbar_label = field
            extra_info = None

        fig, ax = self._plot_masked_map(
            data, mask, cmap=cmap, title=title, cbar_label=cbar_label,
            cbar_ticks=cbar_ticks, extra_coord_info=extra_info
        )
        plt.show()

    # ---------- other plotting methods ----------
    def plot_projection(self, method: str = 'mean', smoothed: bool = False) -> None:
        """Plot a 2D projection of the volume."""
        vol = self.calc.smoothed_volume if smoothed else self.calc.volume
        proj = getattr(vol, method)(axis=0)
        title = f"2D Projection ({method}, {'smoothed' if smoothed else 'raw'})"
        self._plot_masked_map(
            proj, None, cmap='gray', title=title, cbar_label='Density'
        )
        plt.show()

    def interactive_slice_viewer(self, smoothed: bool = False, init_z: float = 0.0) -> None:
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
                self._plot_masked_map(
                    vol[idx], None, cmap='gray',
                    title=f"Slice Z={z0:.1f} Å", cbar_label='Density'
                )
                plt.show()

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

        def compute(zmin: float, zmax: float, thresh: float) -> np.ndarray:
            i0, i1 = int(zmin / pix), int(zmax / pix) + 1
            comp = vol[i0:i1].mean(axis=0)
            norm = (comp - comp.min()) / (np.ptp(comp) + 1e-8)
            return (norm > thresh).astype(int)

        self._interactive_mask_creator(
            compute, save_attr='intensity_mask_2d',
            description='Intensity Mask', init_z_range=init_z_range,
            init_thresh=init_thresh, plot=plot
        )

    def interactive_mask_collapse(self,
                                  init_z_range: Optional[Tuple[float, float]] = None,
                                  init_thresh: float = 0.99,
                                  plot: bool = True) -> None:
        """Interactive projection of 3D mask to 2D via thresholding."""
        mask3d = self.calc.mask3d
        assert mask3d is not None, "Load a 3D mask first."
        pix = self.calc.pixel_size

        def compute(zmin: float, zmax: float, thresh: float) -> np.ndarray:
            i0, i1 = int(zmin / pix), int(zmax / pix) + 1
            collapsed = ~np.any(mask3d[i0:i1] < thresh, axis=0)
            cy, cx = np.array(collapsed.shape) // 2
            hy, hx = np.array(self.calc.volume.shape[1:]) // 2
            return collapsed[cy-hy:cy-hy+self.calc.volume.shape[1],
                             cx-hx:cx-hx+self.calc.volume.shape[2]]

        self._interactive_mask_creator(
            compute, save_attr='mask_2d',
            description='Mask 2D', init_z_range=init_z_range,
            init_thresh=init_thresh, plot=plot
        )

    def create_combined_mask(self) -> None:
        """Combine projection and intensity masks into fitting_mask."""
        m2d = self.calc.mask_2d
        im2d = self.calc.intensity_mask_2d
        assert m2d is not None and im2d is not None, "Create both masks first."
        if m2d.shape != im2d.shape:
            raise ValueError("Mask shapes mismatch.")
        self.calc.fitting_mask = (m2d & im2d).astype(int)
        self._plot_masked_map(
            self.calc.fitting_mask, None, cmap='gray',
            title="Combined Fitting Mask", cbar_label='Mask'
        )
        plt.show()

    def plot_profile_point(self,
                           x_angs: float,
                           y_angs: float,
                           template_idx: Optional[int] = None,
                           full_range: bool = False,
                           corr_length: float = 6/1.7741,
                           reg_lambda: float = 0.0) -> None:
        """
        Plot intensity profile at (x,y) in Å using the batched GPU fitter,
        under the 4-Gaussian reparam scheme.
        """
        calc = self.calc
        pix  = calc.pixel_size

        x_pix, y_pix = round(x_angs/pix), round(y_angs/pix)
        prof   = calc.smoothed_volume[:, y_pix, x_pix].astype(float)
        zdim   = prof.size
        z_full = np.arange(zdim) * pix

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

            params_b, covs_b, ssr_b, dof_array = fit_batch_profile_gpu(
                seg[np.newaxis,:],
                init_abs, lb_abs, ub_abs,
                i0, pix, corr_length, reg_lambda
            )

            p   = params_b[0]
            ssr = ssr_b[0]
            dof = dof_array[0]

            theta = lb_abs + (ub_abs - lb_abs) * torch.sigmoid(torch.from_numpy(p).to('cpu')).detach().numpy()
            phi = theta_to_phi(theta, pix, i0)
            m1_px, s1_px, A1, m2_px, s2_px, A2, m3_px, s3_px, A3, m4_px, s4_px, A4, bias = phi

            m1 = (m1_px + i0) * pix
            m2 = (m2_px + i0) * pix
            m3 = (m3_px + i0) * pix
            m4 = (m4_px + i0) * pix
            sigma1 = s1_px * pix
            sigma2 = s2_px * pix
            sigma3 = s3_px * pix
            sigma4 = s4_px * pix
            thickness = m2 - m1
            rmse = np.sqrt(ssr / dof) if dof > 0 else np.inf

            print(f"Fit at ({x_angs:.1f}Å, {y_angs:.1f}Å), tpl={t_idx}:")
            print(f"  Peak1: z={m1:.2f}Å, σ={sigma1:.2f}Å, A={A1:.2f}")
            print(f"  Peak2: z={m2:.2f}Å, σ={sigma2:.2f}Å, A={A2:.2f}")
            print(f"  Peak3: z={m3:.2f}Å, σ={sigma3:.2f}Å, A={A3:.2f}")
            print(f"  Peak4: z={m4:.2f}Å, σ={sigma4:.2f}Å, A={A4:.2f}")
            print(f"  Offset={bias:.2f}; Thickness={thickness:.2f}Å; RMSE={rmse:.4f}")

            plt.figure(figsize=(6,4))
            ax = plt.gca()
            self._plot_gaussian_profile(
                ax, prof, i0, i1, pix, phi, full_range
            )
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
            ax: Optional[plt.Axes] = None,
            *,
            verbose: bool = True,
            annotate_thickness: bool = False,
            thickness_label_fontsize: int = 9,
            thickness_label_bbox: Optional[dict] = None,
    ) -> Optional[plt.Axes]:
        """
        Plot the profile using precomputed θ from compute_thickness, without refitting.
        """
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
            created_fig = True

        calc = self.calc
        pix = calc.pixel_size

        i = round(y_angs / pix)
        j = round(x_angs / pix)

        t_idx = calc.template_map[i, j]
        if np.isnan(t_idx) or t_idx < 0 or t_idx >= len(calc.fit_templates):
            if verbose:
                print(f"No fit stored at ({x_angs:.1f}Å,{y_angs:.1f}Å)")
            return ax
        t_idx = int(t_idx)

        theta = calc.theta_map[i, j, :]
        if np.isnan(theta).any():
            if verbose:
                print(f"No fit stored at ({x_angs:.1f}Å,{y_angs:.1f}Å)")
            return ax

        _, _, _, (z_lo, z_hi) = calc.fit_templates[t_idx]
        i0 = round(z_lo / pix)
        i1 = round(z_hi / pix)

        prof = calc.smoothed_volume[:, i, j].astype(float)
        phi = theta_to_phi(theta, pix, i0)

        m1_px, s1_px, A1, m2_px, s2_px, A2, m3_px, s3_px, A3, m4_px, s4_px, A4, bias = phi
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

        self._plot_gaussian_profile(ax, prof, i0, i1, pix, phi, full_range)

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