import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import copy

from fitting import *

def plot_chi2_slice(
        chi2_grid: np.ndarray,
        param1_vals: np.ndarray,
        param2_vals: np.ndarray,
        param1_name: str,
        param2_name: str,
        obj_name='obt_name',
        other_axes: dict = None,
        param_labels: dict = None

):
    """
    Plot the 2D chi-squared grid for any pair of parameters by minimizing over the other axes.

    other_axes: dict, e.g. { 'rK': idx_rK, 'vsini': idx_vsini, 'B': idx_B }
        If specified, slices those axes at the given indices instead of minimizing.
    param_labels: dict, label overrides for pretty plot titles.
    """
    if param_labels is None:
        param_labels = {}
    # Map from names to axis indices in the grid
    param_axes = ['Teff', 'logg', 'rK', 'vsini', 'B']
    param_idxs = {k: i for i, k in enumerate(param_axes)}

    idx1 = param_idxs[param1_name]
    idx2 = param_idxs[param2_name]

    # Axes to minimize/slice over
    axes = list(range(chi2_grid.ndim))
    axes.remove(idx1)
    axes.remove(idx2)
    min_axes = axes
    # If you want to slice (fix) some axes instead of minimizing, you can do so
    if other_axes:
        # Build slices
        slicer = [slice(None)] * chi2_grid.ndim
        for k, v in other_axes.items():
            slicer[param_idxs[k]] = v
        reduced_chi2 = chi2_grid[tuple(slicer)]
    else:
        # Profile/minimize over other parameters
        reduced_chi2 = np.min(chi2_grid, axis=tuple(min_axes))

    # Make the plot
    plt.figure(figsize=(7, 6))
    CS = plt.contourf(param1_vals, param2_vals, reduced_chi2.T, levels=30, cmap='viridis')
    plt.colorbar(CS, label='Chi-squared')
    plt.xlabel(param_labels.get(param1_name, param1_name))
    plt.ylabel(param_labels.get(param2_name, param2_name))
    plt.title(
        f'Chi-squared: {param_labels.get(param1_name, param1_name)} vs {param_labels.get(param2_name, param2_name)}')
    plt.scatter(
        param1_vals[np.argmin(np.min(reduced_chi2, axis=1))],
        param2_vals[np.argmin(np.min(reduced_chi2, axis=0))],
        marker='*', color='red', s=120, label='Minimum'
    )
    plt.legend()
    plt.tight_layout()
    # plt.savefig(
    #     f"{outdir}/{obj_name}_Regions{regions}_chi2_Teff{Teff_best}_logg{logg_best}_rK{round(rK_best,2)}_vsini{vsini_best}_B{B_best}.png",
    #     dpi=120, facecolor="white")
    plt.show()


def plot_1d_chi2_profile(param_vals, chi2_grid, axis, param_name, delta_chi2=1.0):
    # Profile over all other parameters
    chi2_profile = np.min(chi2_grid, axis=tuple(a for a in range(chi2_grid.ndim) if a != axis))
    min_chi2 = np.min(chi2_profile)
    threshold = min_chi2 + delta_chi2

    plt.figure(figsize=(6,4))
    plt.plot(param_vals, chi2_profile, label=f'χ² profile: {param_name}')
    plt.axhline(threshold, color='orange', linestyle='--', label=f'χ² min + {delta_chi2}')
    plt.axvline(param_vals[np.argmin(chi2_profile)], color='red', linestyle=':', label='Best fit')
    plt.xlabel(param_name)
    plt.ylabel('Chi-squared')
    plt.title(f'Profile χ² for {param_name}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Report confidence interval
    ok = (chi2_profile - min_chi2) <= delta_chi2
    if np.any(ok):
        conf_interval = (param_vals[ok].min(), param_vals[ok].max())
    else:
        conf_interval = (None, None)
    print(f'Best fit {param_name}: {param_vals[np.argmin(chi2_profile)]}')
    print(f'{delta_chi2:.2f} delta chi² confidence interval: {conf_interval}')
    return conf_interval


def plot_bestfit_and_residuals(
        data,
        best_params,
        regions,
        MoogStokesModel=MoogStokesModel,
        shifts=None,
        renormalization=None,
        obj_name="object",
        outdir="figures/model_plots",
        ylim=(0.55, 1.1),
):
    """
    Plot the data, best-fit model, and residuals (with errors) for each region.

    Parameters
    ----------
    data : ProplydData
        Observed data object.
    best_params : tuple
        (Teff, logg, rK, vsini, B)
    regions : list[int]
        List of region indices.
    MoogStokesModel : class
        Model class (default: MoogStokesModel).
    obj_name : str
        Name of object for figure title/output.
    outdir : str
        Output directory for saving the figure.
    ylim : tuple
        y-limits for spectra panels.
    """
    Teff_best, logg_best, rK_best, vsini_best, B_best = best_params


    n_regions = len(regions)
    nrows = n_regions

    if shifts is None:
        shifts = np.zeros(n_regions)

    if renormalization is None:
        renormalization = np.ones(n_regions)

    fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=(10, 2.2 * nrows), gridspec_kw={"width_ratios": [3, 2]})

    for idx, r in enumerate(regions):
        # Get data in this region

        shift=shifts[idx]
        data.doppler_shift_data(shift)

        xlo, xhi = MoogStokesModel.region_xlims(r)
        xdata, ydata, yerrdata = data.get_range(xlo, xhi)
        ydata,yerrdata = ydata*renormalization[idx],yerrdata*renormalization[idx]
        model = MoogStokesModel(Teff_best, logg_best, rK_best, B_best, vsini_best, r)
        ymodel = model.interpolate(xdata)

        # Spectra panel
        ax = axs[idx, 0]
        ax.plot(xdata, ydata, c='midnightblue', label='Data')
        ax.plot(xdata, ymodel, c='magenta', label='Model')
        ax.fill_between(xdata, ydata - yerrdata, ydata + yerrdata, color='blue', alpha=0.2, lw=0)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(*ylim)
        ax.set_ylabel("Flux")
        if idx == 0:
            ax.legend(fontsize=9)
        ax.set_title(f"Region {r}")
        if idx == nrows - 1:
            ax.set_xlabel("Wavelength (Å)")

        # Residuals panel
        ax_res = axs[idx, 1]
        residual = ydata - ymodel
        ax_res.axhline(0, color='gray', lw=1)
        ax_res.plot(xdata, residual, c='teal', label='Residual')
        ax_res.fill_between(xdata, -yerrdata, yerrdata, color='orange', alpha=0.2, lw=0, label='±1σ error')
        ax_res.set_xlim(xlo, xhi)
        ax_res.set_ylabel("Residuals")
        # Visual scale: residuals, error
        ax_res.set_ylim(-4.5 * np.nanmedian(yerrdata), 4.5 * np.nanmedian(yerrdata))
        if idx == 0:
            ax_res.legend(fontsize=9)
        if idx == nrows - 1:
            ax_res.set_xlabel("Wavelength (Å)")

        data.doppler_shift_data(-shift)

    fig.suptitle(
        f"{obj_name}: Teff={Teff_best}, logg={logg_best}, rK={round(rK_best,2)}, vsini={vsini_best}, B={B_best}",
        fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    # Save if outdir is provided
    import os
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(
        f"{outdir}/{obj_name}_Regions{regions}_Teff{Teff_best}_logg{logg_best}_rK{round(rK_best,2)}_vsini{vsini_best}_B{B_best}.png",
        dpi=120, facecolor="white")
    plt.show()

def plot_data_and_model(obj: ProplydData,
                        Teff: float, logg: float, rK: float, B: float, vsini: float,
                        ylim: tuple | None = None,
                        renormalization: NDArray | None = None,
                        shifts: NDArray | None = None,
                        fname: str | None = None,
                        label_regions: bool = True,
                       ):
    """Plots data together with a test MoogStokes model. Helpful for setting
    renormalization and shifts for each region before fitting, or for viewing
    the best-fit model.

    Parameters
    ----------
    obj: ProplydData
        Contains the target's spectrum.
    Teff, logg, rK, B, vsini: float
        MoogStokes model parameters.
    ylim: tuple of floats or None
        If not None, sets the y-axis of the plot to (ylim[0], ylim[1]).
    renormalization: array_like of shape (Nreg,)
        Renormalization for each region. Nreg is the number of regions
        defined by the class attribute MoogStokesModel.NUM_REGIONS.
    shifts: array_like of shape (Nreg, )
        Shift of the data relative to the model for each region in pixels.
    fname: str or None
        If not None, saves the figure to the file name.

    Returns
    -------
    None
    """
    Nreg = MoogStokesModel.NUM_REGIONS
    if ylim is None:
        ylim = (0.75, 1.05)
    if renormalization is None:
        renormalization = np.ones(Nreg)
    if shifts is None:
        shifts = np.zeros(Nreg)
    
    fig, axs =setup_regions_plot(label_regions=label_regions)
    for r, ax in enumerate(axs):
        obj_plot = copy.deepcopy(obj)
        obj_plot.doppler_shift_data(shifts[r])
        model = MoogStokesModel(Teff=Teff, logg=logg, rK=rK, B=B, vsini=vsini, region=r)
        renorm = renormalization[r]
        ax.plot(obj_plot.x, obj_plot.y * renorm, c='midnightblue')
        ax.plot(model.x, model.y, c='magenta')
        ax.set_ylim(ylim)
        
    fig.suptitle(f"{obj.name}: Teff={model.Teff}, logg={model.logg}, rK={round(model.rK,2)}, vsini={model.vsini}, B={model.B}")

    if fname:
        fig.savefig(fname, dpi=300, facecolor="white")
    
    plt.show()
    return fig, axs

def setup_regions_plot(label_regions=True):
    """
    Returns
    -------
    fig, axs : matplotlib Figure and Axes
        Figure and Axes objects for the regions plot.
    """
    Nreg = MoogStokesModel.NUM_REGIONS
    Nsub = Nreg  # number of subplots
    if Nreg % 2 != 0:
        Nsub += 1  ## need
    
    fig, axs = plt.subplots(nrows=Nsub//2, ncols=2, figsize=(8.5,7))
    axs = axs.reshape(-1)

    for r, ax in enumerate(axs):
        if (Nreg % 2 != 0) and r == Nsub - 1:
            ax.axis("off")
            continue
        ax.set_ylim(0.6, 1.05)
        xlo, xhi = MoogStokesModel.region_xlims(r)
        ax.set_xlim(xlo, xhi)

        if label_regions:
            ax.text(0.81, 0.05, f"Region {r}", transform=ax.transAxes, color='k')
        
    fig.supxlabel(r"Wavelength ($\AA$)")
    fig.supylabel(r"Normalized Flux Density")
    plt.tight_layout()

    if Nreg % 2 != 0: # If an odd number, remove the last subplot
        axs_new = axs[:-1]
    else:
        axs_new = axs
    return fig, axs_new

def setup_regions_plot_equalx(show_regions=True, valid_regions=range(7)):
    """Each subplot is the same size in wavelength, with the fitted regions
    highlighted. More similar to what Christian usually has in his papers.

    Parameters
    ----------
    show_regions: bool
        If True, shades in the fitted regions in green.
    valid_regions:
        Regions not included in this list (invalid regions) are shaded in red.
        Ignored if show_regions = False.

    Returns
    -------
    fig, axs : matplotlib Figure and Axes
        Figure and Axes objects for the regions plot.
    """
    PLOT_XSIZE = 240 # angstroms
    PLOT_XCENTERS = (
        21120, # regions 0 and 1
        21850, # regions 2 and 3
        22080, # region 4 (Na)
        22250, # region 5
        22635, # region 6 (Ca)
        22980, # region 7 (CO)
    )
    Nsub = len(PLOT_XCENTERS)

    fig, axs = plt.subplots(nrows=Nsub, ncols=1, figsize=(8.5,7))
    axs = axs.reshape(-1)

    for i, ax in enumerate(axs):

        xlo = PLOT_XCENTERS[i] - PLOT_XSIZE / 2
        xhi = PLOT_XCENTERS[i] + PLOT_XSIZE / 2
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(0.6, 1.05)

        if show_regions:
            for n in range(MoogStokesModel.NUM_REGIONS):
                xlo, xhi = MoogStokesModel.region_xlims(n)
                if n in valid_regions:
                    ax.axvspan(xlo, xhi, color='green', alpha=0.2)
                else:
                    ax.axvspan(xlo, xhi, color='red', alpha=0.4)
                
    fig.supxlabel(r"Wavelength ($\AA$)", fontsize=12)
    fig.supylabel(r"Normalized Flux Density", fontsize=12)
    plt.tight_layout()

    return fig, axs