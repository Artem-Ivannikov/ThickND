import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import rotate as nd_rotate

# ------------------------------------------------------------
#  Helper: Å tick formatter
# ------------------------------------------------------------
def _angstrom_formatter():
    def fmt(x, pos):
        if abs(x - round(x)) < 1e-6:
            return f"{int(round(x))} Å"
        if abs(x * 10 - round(x * 10)) < 1e-6:
            return f"{x:.1f} Å"
        return f"{x:.2f} Å"
    return FuncFormatter(fmt)


# ------------------------------------------------------------
#  Basic image processing (kept as in original but cleaned)
# ------------------------------------------------------------
def filter_thickness(width, width_sigma, sigma_thresh):
    result = width.astype(float).copy()
    result[width_sigma > sigma_thresh] = np.nan
    return result

def mirror_image(img, axis):
    if axis.lower() == 'x':
        return img[:, ::-1]
    elif axis.lower() == 'y':
        return img[::-1, :]
    raise ValueError(f"Invalid axis '{axis}'. Choose 'x' or 'y'.")

def rotate_image(img, angle, reshape=False):
    return nd_rotate(img, angle, reshape=reshape, order=1, mode='constant', cval=np.nan)

def crop_image(img, x_range, y_range, pixel_size):
    x_min, x_max = x_range
    y_min, y_max = y_range
    ix0 = int(np.floor(x_min / pixel_size))
    ix1 = int(np.ceil(x_max / pixel_size))
    iy0 = int(np.floor(y_min / pixel_size))
    iy1 = int(np.ceil(y_max / pixel_size))
    iy0 = max(0, iy0)
    iy1 = min(img.shape[0], iy1)
    ix0 = max(0, ix0)
    ix1 = min(img.shape[1], ix1)
    return img[iy0:iy1, ix0:ix1]

def process_thickness_maps(width, width_sigma, sigma_thresh,
                           mirror_axis=None, rotation_angle=None,
                           x_range=None, y_range=None, pixel_size=1.0,
                           reshape_rotate=False):
    w = filter_thickness(width, width_sigma, sigma_thresh)
    ws = width_sigma.astype(float).copy()
    ws[np.isnan(w)] = np.nan
    if mirror_axis:
        w = mirror_image(w, mirror_axis)
        ws = mirror_image(ws, mirror_axis)
    if rotation_angle is not None:
        w = rotate_image(w, rotation_angle, reshape=reshape_rotate)
        ws = rotate_image(ws, rotation_angle, reshape=reshape_rotate)
    if x_range is not None and y_range is not None:
        w = crop_image(w, x_range, y_range, pixel_size)
        ws = crop_image(ws, x_range, y_range, pixel_size)
    return w, ws


# ------------------------------------------------------------
#  Plotting primitives (all accept an `ax` argument)
# ------------------------------------------------------------
def plot_line_on_map(img, pixel_size, point1, point2, ax,
                     colorbar=True, cbar_label=None, **imshow_kwargs):
    """
    Plot image and overlay line between two points (Å).
    """
    imshow_kwargs.setdefault("cmap", "cividis")
    extent = [0, img.shape[1] * pixel_size, 0, img.shape[0] * pixel_size]
    im = ax.imshow(img, extent=extent, origin='lower', **imshow_kwargs)

    x1, y1 = point1
    x2, y2 = point2
    ax.plot([x1, x2], [y1, y2], color='red', linewidth=1.75, zorder=5)
    ax.plot([x1, x2], [y1, y2], 'o', color='red', markersize=5, zorder=6)

    if colorbar:
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        if cbar_label:
            cbar.set_label(cbar_label)
        cbar.ax.yaxis.set_major_formatter(_angstrom_formatter())
    return ax


def extract_line_profile(img, pixel_size, point1, point2, num=100):
    x1, y1 = point1
    x2, y2 = point2
    xp1, yp1 = x1/pixel_size, y1/pixel_size
    xp2, yp2 = x2/pixel_size, y2/pixel_size
    xs = np.linspace(xp1, xp2, num)
    ys = np.linspace(yp1, yp2, num)
    xi = np.clip(np.round(xs).astype(int), 0, img.shape[1]-1)
    yi = np.clip(np.round(ys).astype(int), 0, img.shape[0]-1)
    values = img[yi, xi]
    distances = np.sqrt((xs-xp1)**2 + (ys-yp1)**2) * pixel_size
    return distances, values


def plot_line_profile(distances, values, ax, sigma=None,
                      ylabel='Thickness (Å)', title=None):
    """
    Plot 1D profile with optional ±1σ shading.
    """
    if sigma is not None:
        ax.fill_between(distances, values - sigma, values + sigma,
                        color='gray', alpha=0.3, label='±1σ')
    ax.plot(distances, values, '-', label='Profile')
    ax.set_xlabel('Distance (Å)')
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True)
    return ax


# ------------------------------------------------------------
#  Inverse transform (map processed coords back to original)
# ------------------------------------------------------------
def inverse_transform_coordinate(x_final, y_final, original_shape, pixel_size,
                                 mirror_axis=None, rotation_angle=None,
                                 x_range=None, y_range=None, reshape_rotate=False):
    x = x_final
    y = y_final

    # Undo crop
    if x_range is not None and y_range is not None:
        x += int(np.floor(x_range[0] / pixel_size)) * pixel_size
        y += int(np.floor(y_range[0] / pixel_size)) * pixel_size

    # Undo rotation
    if rotation_angle is not None:
        x_pix = x / pixel_size
        y_pix = y / pixel_size
        h, w = original_shape
        cx, cy = w/2, h/2
        angle_rad = np.radians(rotation_angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        x_centered = x_pix - cx
        y_centered = y_pix - cy
        x_rot = x_centered * cos_a - y_centered * sin_a
        y_rot = x_centered * sin_a + y_centered * cos_a
        x_pix = x_rot + cx
        y_pix = y_rot + cy
        x = x_pix * pixel_size
        y = y_pix * pixel_size

    # Undo mirror
    if mirror_axis:
        h, w = original_shape
        if mirror_axis.lower() == 'x':
            x = w * pixel_size - x
        elif mirror_axis.lower() == 'y':
            y = h * pixel_size - y
    return x, y



def make_panel(calc,
               p0, p1,
               profile_points,
               viewer,
               sigma_thresh=5.0,
               mirror_axis='y',
               rotation_angle=0.0,
               x_range=None,
               y_range=None,
               reshape_rotate=False,
               cmap_map='cividis',
               color_cycle='Dark2',
               figsize=(16, 9),
               dpi=120,
               left_right_widths=(1.2, 1.8),
               left_height_fracs=(0.62, 0.28),
               max_per_col=3,
               profile_cols=None,
               suppress_profile_legends=True,
               suppress_profile_titles=True,
               annotate_profile_numbers=False,
               mark_profiles_on_map=True,
               map_marker_style='X',
               map_marker_size=12,
               map_marker_linewidth=2.0,
               title_fontsize=16,
               label_fontsize=14,
               tick_labelsize=11,
               profile_spine_linewidth=1.6,
               profile_thickness_fontsize=11,
               thickness_legend_fontsize=12,
               savepath=None,
               savefig_dpi=300,
               show=True):
    """
    Build the two‑column thickness panel directly from calc, processing maps internally.

    Parameters
    ----------
    calc : object
        Must have attributes: pixel_size, width, width_sigma, smoothed_volume,
        theta_map, template_map, fit_templates.
    p0, p1 : tuple of floats
        Line endpoints (Å) for thickness profile.
    profile_points : list of (x,y)
        Coordinates (Å) for fit profiles.
    viewer : object
        Must have method plot_profile_from_map(x, y, ax=..., ...)
    sigma_thresh : float
        Threshold for width_sigma filtering.
    mirror_axis, rotation_angle, x_range, y_range : processing parameters
        applied to the thickness maps before display.
    reshape_rotate : bool
        Whether rotation uses reshape=True (usually False).
    ... (rest of visual parameters as before)
    """
    pixel_size = calc.pixel_size
    if pixel_size is None:
        raise ValueError("calc.pixel_size missing")

    # --- Process thickness maps internally ---
    w, ws = process_thickness_maps(
        calc.width, calc.width_sigma,
        sigma_thresh=sigma_thresh,
        mirror_axis=mirror_axis,
        rotation_angle=rotation_angle,
        x_range=x_range, y_range=y_range,
        pixel_size=pixel_size,
        reshape_rotate=reshape_rotate
    )

    n_profiles = len(profile_points)
    if n_profiles < 1:
        raise ValueError("profile_points must be non‑empty")

    # Grid layout
    n_cols = profile_cols if profile_cols else max(1, math.ceil(n_profiles / max_per_col))
    n_rows = math.ceil(n_profiles / n_cols)

    t_frac, b_frac = left_height_fracs
    height_ratios = [t_frac / (t_frac + b_frac), b_frac / (t_frac + b_frac)]
    left_w, right_w = left_right_widths

    # Colors for profile outlines
    cmap_prof = plt.get_cmap(color_cycle)
    base_colors = [cmap_prof(i) for i in range(cmap_prof.N)]

    # Local rc settings
    rc = {
        "figure.dpi": dpi,
        "axes.titlesize": title_fontsize,
        "axes.labelsize": label_fontsize,
        "xtick.labelsize": tick_labelsize,
        "ytick.labelsize": tick_labelsize,
        "grid.linewidth": 0.6,
    }

    angfmt = _angstrom_formatter()

    with plt.rc_context(rc):
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2,
                      width_ratios=[left_w, right_w],
                      height_ratios=height_ratios,
                      left=0.05, right=0.98, bottom=0.06, top=0.95,
                      wspace=0.22, hspace=0.22)

        ax_map = fig.add_subplot(gs[0, 0])
        ax_thick = fig.add_subplot(gs[1, 0])
        gs_right = GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=gs[:, 1],
                                           hspace=0.28, wspace=0.26)

        # Create profile axes column-major
        ax_profiles = [fig.add_subplot(gs_right[r, c])
                       for c in range(n_cols) for r in range(n_rows)]

        # ---- Left top: thickness map ----
        plot_line_on_map(w, pixel_size, p0, p1, ax_map, cmap=cmap_map)
        ax_map.set_title("Thickness map")
        ax_map.xaxis.set_major_formatter(angfmt)
        ax_map.yaxis.set_major_formatter(angfmt)
        ax_map.set_xlabel("")
        ax_map.set_ylabel("")

        # ---- Left bottom: thickness profile ----
        d, v = extract_line_profile(w, pixel_size, p0, p1)
        sigma_line = None
        if ws is not None:
            _, sigma_line = extract_line_profile(ws, pixel_size, p0, p1)

        plot_line_profile(d, v, ax_thick, sigma=sigma_line,
                          ylabel='Thickness (Å)', title="Thickness profile")
        patch = Patch(facecolor='gray', alpha=0.3, edgecolor='none', label='±1σ')
        ax_thick.legend(handles=[patch], loc='upper right',
                        framealpha=0.95, fontsize=thickness_legend_fontsize)
        ax_thick.xaxis.set_major_formatter(angfmt)
        ax_thick.yaxis.set_major_formatter(angfmt)
        ax_thick.set_xlabel("Distance (Å)")
        ax_thick.set_ylabel("Thickness (Å)")

        # ---- Right: fit profiles ----
        profile_meta = []
        for idx, (xp, yp) in enumerate(profile_points):
            if idx >= len(ax_profiles):
                break
            ax = ax_profiles[idx]
            color = base_colors[idx % len(base_colors)]

            # Convert map coordinate back to original (unprocessed) image coordinate
            xv, yv = inverse_transform_coordinate(
                xp, yp, calc.width.shape,
                pixel_size=pixel_size,
                mirror_axis=mirror_axis,
                rotation_angle=rotation_angle,
                x_range=x_range, y_range=y_range
            )

            # Plot stored fit profile
            viewer.plot_profile_from_map(
                xv, yv,
                full_range=False,
                ax=ax,
                verbose=False,
                annotate_thickness=True,
                thickness_label_fontsize=profile_thickness_fontsize,
                thickness_label_bbox=None
            )

            # Colored spine to link with map
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(profile_spine_linewidth)

            # Formatting
            ax.xaxis.set_major_formatter(angfmt)
            ax.set_xlabel("")
            ax.set_ylabel("")

            if annotate_profile_numbers:
                ax.text(0.02, 0.95, f"{idx+1}",
                        transform=ax.transAxes,
                        ha='left', va='top', fontsize=10,
                        color='white',
                        bbox=dict(boxstyle='circle,pad=0.2',
                                  fc=color, ec='none', alpha=0.95))

            if suppress_profile_titles:
                ax.set_title("")
            if suppress_profile_legends:
                leg = ax.get_legend()
                if leg:
                    leg.remove()

            profile_meta.append({
                "index": idx,
                "map_coord": (xp, yp),
                "viewer_coord": (xv, yv),
                "color": color,
                "axis": ax
            })

        # Hide unused axes
        for ax in ax_profiles[n_profiles:]:
            ax.set_visible(False)

        # Mark profile points on map
        if mark_profiles_on_map:
            for info in profile_meta:
                xp, yp = info["map_coord"]
                color = info["color"]
                ax_map.plot(xp, yp, marker=map_marker_style,
                            markersize=map_marker_size,
                            markeredgewidth=map_marker_linewidth,
                            color=color, linestyle='None', zorder=12)

        # Tick size
        ax_map.tick_params(labelsize=tick_labelsize)
        ax_thick.tick_params(labelsize=tick_labelsize)
        for info in profile_meta:
            if info["axis"].get_visible():
                info["axis"].tick_params(labelsize=tick_labelsize)

        if savepath:
            fig.savefig(savepath, dpi=savefig_dpi, bbox_inches='tight')
        if show:
            plt.show()

    meta = {
        "map_axis": ax_map,
        "thickness_axis": ax_thick,
        "profile_axes": [m["axis"] for m in profile_meta],
        "profile_meta": profile_meta,
        "profile_grid_shape": (n_rows, n_cols),
    }
    return fig, meta