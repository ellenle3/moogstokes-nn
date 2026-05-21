import matplotlib.pyplot as plt
import time
import os
import copy
import warnings
import numpy as np

from numpy.typing import NDArray
from scipy.interpolate import interp1d
from spectra import SpectralData, MoogStokesModel, BTSettlModel, setup_regions_plot
from astropy.io import fits
from interpolate_moogstokes import interpolate_moogstokes_models
from nn_helpers import MoogStokesNN

def chi_squared(ydata: NDArray, yerrdata: NDArray, ymodel: NDArray) -> float:
    """Calculate the chi-squared statistic for the given data and model spectrum.
    All arrays must be the same length.
    """
    return np.nansum((ydata-ymodel)**2/yerrdata**2)

def compute_moogstokes_chi2_grid(data: SpectralData, Teff_vals: NDArray, logg_vals:
                                 NDArray, rK_vals: NDArray, vsini_vals: NDArray,
                                 B_vals: NDArray, renormalization: NDArray | None = None,
                                 shifts: NDArray | None = None, regions: list[int] | None = None,
                                 resolution: float | None = None, kernel: str | None = None) -> NDArray:
    """Computes the chi-squared statistic across a grid of MoogStokes models.
    The models are interpolated to match the x-values of the data.

    Bear in mind that this function can become excruciatingly slow if the grid
    of parameters is too large.

    Parameters
    ----------
    data: SpectralData
        Science data to fit.
    Teff_vals: NDArray
        Effective temperatures across the grid.
    ...
    regions: list of int or None, optional
        Wavelength regions of MoogStokes models to fit. If None, all regions (0
        through 6) will be used.
    
    renormalization: list of floats, optional
        Small changes to the normalization values that can be applied to each region
    shifts: NDArray, optional
        Shifts in pixels applied to each region. Due to innacuracies in wavelength position
        of lines, and wavelength calibration of data we expect some pixel level shifts.
    Returns
    -------
    chi2: NDArray
        Chi-squared statistic across the grid of parameters indexed in the order
        (Teff, logg, rK, vsini, B).
    """
    if regions is None:
        regions = range(7)

    if shifts is None:
        shifts = np.zeros(len(regions))

    if renormalization is None:
        renormalization = np.ones(len(regions))

    chi2 = np.zeros( (len(Teff_vals), len(logg_vals), len(rK_vals), len(vsini_vals), len(B_vals)) )
    chi2_region = np.zeros( (len(Teff_vals), len(logg_vals), len(rK_vals), len(vsini_vals), len(B_vals)) )


    for nn, r in enumerate(regions):

        data.doppler_shift_data(shifts[nn])

        xlo, xhi = MoogStokesModel.region_xlims(r)
        xdata, ydata, yerrdata = data.get_range(xlo, xhi)

        #### Apply the desired normalization
        ydata, yerrdata = ydata*renormalization[nn] , yerrdata*renormalization[nn]

        xdata = np.array(xdata)
        ydata = np.array(ydata)
        yerrdata = np.array(yerrdata)

        # ####### THIS BLOCK NEEDS TO BE COMMENTED OUT ######
        #
        # rng = np.random.default_rng(42)
        # noise = rng.normal(loc=0.0, scale=1 / 100., size=len(ydata))
        # ydata = ydata + noise
        # mean_snr = np.nanmean(yerrdata)
        # yerrdata = yerrdata * 100 ** (-1) / mean_snr

        #print(np.nanmean(yerrdata))

        for i, Teff in enumerate(Teff_vals):
            for j, logg in enumerate(logg_vals):
                for k, rK in enumerate(rK_vals):
                    for l, vsini in enumerate(vsini_vals):
                        for m, B in enumerate(B_vals):

                            try:
                                model = MoogStokesModel(Teff, logg, rK, B, vsini, r)
                            except FileNotFoundError:
                                print("here")
                                chi2_region[i, j, k, l, m] = np.nan
                                continue
                            
                            if resolution is not None or kernel is not  None:
                                model.resolution_change(resolution=resolution, Kernel=kernel)

                            ymodel = model.interpolate(xdata)
                            ymodel = np.array(ymodel)



                            chi2_region[i, j, k, l, m] = chi_squared(ydata, yerrdata, ymodel)

        ### sum chi_squared of different regions
        chi2 = chi2 + chi2_region
        chi2_region = np.zeros((len(Teff_vals), len(logg_vals), len(rK_vals), len(vsini_vals), len(B_vals)))

        ### get spectrum back to original position so another shift can be applied to next region

        data.doppler_shift_data(-shifts[nn])
        #print(renormalization[nn])
    return chi2

def best_chi2_grid_params(chi2: NDArray, Teff_vals: NDArray, logg_vals: NDArray,
                          rK_vals: NDArray, vsini_vals: NDArray, B_vals: NDArray) -> tuple[float, float, float, float, float]:
    """Returns the best-fit parameters (minimum chi-squared) from a grid generated
    by compute_moogstokes_chi2_grid().

    Parameters
    ----------
    chi2: NDArray
        Chi-squared statistic across the grid of parameters indexed in the order
        (Teff, logg, rK, vsini, B).
    """
    idxs = np.unravel_index(chi2.argmin(), chi2.shape)
    Teff_best = Teff_vals[idxs[0]]
    logg_best = logg_vals[idxs[1]]
    rK_best = rK_vals[idxs[2]]
    vsini_best = vsini_vals[idxs[3]]
    B_best = B_vals[idxs[4]]
    
    return Teff_best, logg_best, rK_best, vsini_best, B_best


def compute_reduced_chi2_bestfit(
    data,
    best_params,
    regions,
    num_params=5,
    MoogStokesModel=MoogStokesModel
):
    """
    Compute reduced chi-squared for best-fit parameters given data and regions.

    Parameters
    ----------
    data : SpectralData
        The observed data object.
    best_params : tuple
        Best-fit parameters: (Teff, logg, rK, vsini, B)
    regions : list[int]
        List of region indices to stitch.
    num_params : int
        Number of fitted model parameters (default: 5).
    MoogStokesModel : class
        The model class used for generating synthetic spectra.

    Returns
    -------
    reduced_chi2 : float
        The reduced chi-squared value for the best fit.
    """
    Teff_best, logg_best, rK_best, vsini_best, B_best = best_params

    xdata = []
    ydata = []
    yerrdata = []
    ymodel = []

    for r in regions:
        xlo, xhi = MoogStokesModel.region_xlims(r)
        xtemp, ytemp, yerrtemp = data.get_range(xlo, xhi)
        model = MoogStokesModel(Teff_best, logg_best, rK_best, B_best, vsini_best, r)
        ymodeltemp = model.interpolate(xtemp)

        xdata += list(xtemp)
        ydata += list(ytemp)
        yerrdata += list(yerrtemp)
        ymodel += list(ymodeltemp)

    xdata = np.array(xdata)
    ydata = np.array(ydata)
    yerrdata = np.array(yerrdata)
    ymodel = np.array(ymodel)

    # Only use finite (non-nan) values for chi2 calculation
    valid = np.isfinite(ydata) & np.isfinite(yerrdata) & np.isfinite(ymodel)
    n_points = np.sum(valid)
    dof = n_points - num_params

    chi2 = np.nansum((ydata[valid] - ymodel[valid]) ** 2 / yerrdata[valid] ** 2)
    reduced_chi2 = chi2 / dof

    return reduced_chi2

def automatic_wavelength_shifts_values(data: SpectralData, Teff: float, logg:
                                float, rK: float, vsini: float,
                                 B: float, guess_shift: int, regions: list[int] | None = None, use_nn: bool = False) -> NDArray:

    """
    find the best pixel shifts for a specified number of regions for a given model.
    The values of the model are not too important as the lines detected and observed are there.
    :param data: SpectralData
        Observed data object.
    :param Teff: 
    :param logg: 
    :param rK: 
    :param vsini: 
    :param B: 
    :param guess_shift: guess shift to run the chi2 minimzation 
    :param regions: 
    :return: 
    """

    if regions is None:
        regions = range(7)


    best_shift=np.empty(len(regions))

    shift_array=range(-30+guess_shift,30+guess_shift)
    chi2 = np.zeros( (len(regions), len(shift_array)) )

    if use_nn:
        moognn = MoogStokesNN()
    
    for nn, r in enumerate(regions):

        for ii, shifts in enumerate(shift_array):

            data.doppler_shift_data(shifts)

            xlo, xhi = MoogStokesModel.region_xlims(r)
            xdata, ydata, yerrdata = data.get_range(xlo, xhi)

            xdata = np.array(xdata)
            ydata = np.array(ydata)
            yerrdata = np.array(yerrdata)
            
            if use_nn:
                model = moognn.make_moogstokes_model(Teff, logg, rK, B, vsini, r)
            else:
                model = MoogStokesModel(Teff, logg, rK, B, vsini, r)
            ymodel = model.interpolate(xdata)
            ymodel = np.array(ymodel)

            chi2[nn, ii] = chi_squared(ydata, yerrdata, ymodel)

            ### get spectrum back to original position so another shift can be applied to next region
            data.doppler_shift_data(-shifts)


        this_region_min_chi2=np.nanmin(chi2[nn,:])
        best_shift[nn]=shift_array[int(np.nanargmin(chi2[nn,:]))]+guess_shift
        # print(this_region_min_chi2)
        # print(best_shift[nn])

    return best_shift


def get_chi2_confidence_interval(param_vals, chi2_grid, axis, delta_chi2=1.0):
    """
    Computes confidence interval for a single parameter (profiling over the rest).

    param_vals: np.ndarray, grid values for the parameter
    chi2_grid: np.ndarray, chi2 grid
    axis: int, axis of param_vals in chi2_grid
    delta_chi2: float, e.g. 1.0 for 1-sigma, 4.72 for 4 parameters (see table!)
    """
    # Profile over all other parameters (min over the rest)
    chi2_profile = np.min(chi2_grid, axis=tuple(a for a in range(chi2_grid.ndim) if a != axis))
    min_chi2 = np.min(chi2_profile)
    ok = (chi2_profile - min_chi2) <= delta_chi2
    # Return parameter range (can be more than one interval if grid is gappy)
    if np.any(ok):
        return param_vals[ok].min(), param_vals[ok].max()
    else:
        return None, None


def get_all_confidence_intervals(chi2_grid, param_grids, delta_chi2=1.0):
    """
    Returns dict of confidence intervals for each parameter.
    param_grids: dict, e.g., {'Teff': Teff_vals, ...}
    """
    results = {}
    for i, (name, vals) in enumerate(param_grids.items()):
        lo, hi = get_chi2_confidence_interval(vals, chi2_grid, i, delta_chi2=delta_chi2)
        results[name] = (lo, hi)
    return results

def upsample_params(grid_vals, p_best, upsamp_factors, steps, current_iter, interpolation_bounds=None):
    # Normalize p_best to a dict
    if isinstance(p_best, dict):
        p_best_dict = p_best
    else:
        p_best = np.asarray(p_best)
        keys = list(grid_vals.keys())
        if len(p_best) != len(keys):
            raise ValueError("p_best length must match number of parameters")
        p_best_dict = dict(zip(keys, p_best))

    new_grid_vals = {}

    for key, grid in grid_vals.items():
        grid = np.asarray(grid)
        if grid.ndim != 1:
            raise ValueError(f"Grid for parameter {key} must be 1D")

        grid_step_size = grid[1] - grid[0]  # assume uniform spacing
        num_steps = steps[key][current_iter]
        num_steps = abs(num_steps) # force number of steps to be positive
        
        best_val = p_best_dict[key]
        left_val = best_val - num_steps * grid_step_size
        right_val = best_val + num_steps * grid_step_size

        # Apply interpolation bounds if provided
        if interpolation_bounds is not None and key in interpolation_bounds:
            bound_lo, bound_hi = interpolation_bounds[key]
            left_val = max(left_val, bound_lo)
            right_val = min(right_val, bound_hi)

        # Upsample the grid
        new_grid_step_size = grid_step_size / upsamp_factors[key][current_iter]
        new_grid = np.arange(left_val, right_val + new_grid_step_size / 2, new_grid_step_size)

        # Round elements to prevent weird floating point issues
        new_grid = np.round(new_grid, decimals=4)

        new_grid_vals[key] = new_grid

    return new_grid_vals

def write_grid_to_fits(
    dirname: str,
    current_iter: int,
    grid_vals: dict,
    p_best: dict,
    chi2: NDArray,
    renormalization,
    shifts,
    regions,
    data,
    resolution: float | None = None,
    kernel: str | None = None
) -> None:
    """
    Write grid-search results to a FITS file.

    Parameters
    ----------
    fname : str
        Output FITS filename.
    grid_vals : dict
        {param_name: 1D numpy array of grid values}
    p_best : dict
        {param_name: best-fit scalar value}
    chi2, renormalization, shifts : array-like
        Result arrays from the grid search.
    regions : sequence
        Region identifiers.
    data : SpectralData
        Data object containing metadata.
    resolution : float, optional
        Instrument resolution.
    kernel : str, optional
        Kernel name.
    """

    hdus = []

    # Primary HDU: chi2
    primary_hdu = fits.PrimaryHDU(np.asarray(chi2))
    header = primary_hdu.header
    header["TARGET"] = data.name
    header["KERNEL"] = kernel if kernel is not None else "None"
    header["RESPWR"] = resolution if resolution is not None else -1.0
    header["YERSCL"] = data.yerr_scaling
    header["FNDATA"] = data.abs_path
    hdus.append(primary_hdu)

    hdus.append(fits.ImageHDU(np.asarray(renormalization), name="RENORM"))
    hdus.append(fits.ImageHDU(np.asarray(shifts), name="SHIFTS"))
    hdus.append(fits.ImageHDU(np.asarray(regions), name="REGIONS"))

    # Parameter grids: one ImageHDU per parameter
    for key, grid in grid_vals.items():
        hdu = fits.ImageHDU(
            data=np.asarray(grid),
            name=f"GRID_{key}"
        )
        hdus.append(hdu)

    # Best-fit parameters: single-row binary table
    cols = []
    for key, val in p_best.items():
        cols.append(
            fits.Column(
                name=str(key),
                format="D",
                array=[val]
            )
        )

    pbest_hdu = fits.BinTableHDU.from_columns(cols, name="PBEST")
    hdus.append(pbest_hdu)
    
    fname = os.path.join(dirname, f"moogstokes_chi2_iter{current_iter}.fits")
    fits.HDUList(hdus).writeto(fname)

    return os.path.abspath(fname)


def compute_moogstokes_chi2_grid_upsample(data: SpectralData, grid_vals: dict,
                                num_iter: int, upsamp_factors: dict, steps: dict,
                                interpolation_bounds: dict | None = None,
                                renormalization: NDArray | None = None,
                                shifts = None, regions = None,
                                resolution = None, kernel = None, outfile_dir="./"
                                ) -> list[str]:
    
    assert all(all(s > 0 for s in steps[key]) for key in steps), "All steps must be positive integers"
    assert len(next(iter(steps.values()))) == num_iter - 1, "Length of steps must equal num_iter - 1"
    if interpolation_bounds is None:
        warnings.warn("Interpolation bounds are strongly recommended to set a rectangular grid during upsampling.")

    if regions is None:
        regions = range(7)

    if shifts is None:
        shifts = np.zeros(len(regions))

    if renormalization is None:
        renormalization = np.ones(len(regions))
    
    assert len(shifts) == len(regions), "Length of shifts must equal number of regions"
    assert len(renormalization) == len(regions), "Length of renormalizations must equal number of regions"

    fnames = []

    dirname = os.path.join(outfile_dir, data.name)
    if os.path.exists(dirname):
        append_val = 1
        dirname_new = f"{dirname}.{append_val}"
        while os.path.exists(dirname_new) and append_val < 9999:
            dirname_new = f"{dirname}.{append_val}"
            append_val += 1
        if append_val >= 9999:
            raise FileExistsError("Could not create unique output directory name.")
        dirname = dirname_new
    os.makedirs(dirname)

    print(f"Fitting object {data.name}")

    def print_grid_info(param_name, vals):
        dv = vals[1]-vals[0]
        vlo = vals[0]
        vhi = vals[-1]
        print(f"    {param_name}: {len(vals)} steps, {vlo:.4f} to {vhi:.4f}, d{param_name}={dv:.4f}")

    # Create the original grid of models if it doesn't exist yet
    for r in regions:
        interpolate_moogstokes_models(
            models_dir=MoogStokesModel.models_dir,
            output_dir=MoogStokesModel.models_itp_dir,
            region=r,
            target_grids=grid_vals,
            interpolation_bounds=interpolation_bounds
        )

    # Do specified number of steps around the starting region. Get resolution of
    # the original grid
    for i in range(num_iter):
        print(f"Starting iteration {i+1}/{num_iter}. Computing chi-squared across the parameters")
        for key, vals in grid_vals.items():
            print_grid_info(key, vals)
        tstart = time.time()

        Teff_vals = grid_vals["Teff"]
        logg_vals = grid_vals["logg"]
        rK_vals = grid_vals["rK"]
        vsini_vals = grid_vals["vsini"]
        B_vals = grid_vals["B"]
        
        chi2 = compute_moogstokes_chi2_grid(data, Teff_vals, logg_vals, rK_vals,
                                    vsini_vals, B_vals, renormalization,
                                    shifts, regions, resolution, kernel)
        
        Teff, logg, rK, vsini, B = best_chi2_grid_params(chi2, Teff_vals, logg_vals,
                                                        rK_vals, vsini_vals, B_vals)
        
        tend = time.time()
        ttot = tend - tstart
        print(f"Finished grid iteration {i+1}/{num_iter} in {ttot:.2f} seconds." \
              f"    Best-fit params: Teff={Teff:.4f}, logg={logg:.4f}, rK={rK:.4f}, vsini={vsini:.4f}, B={B:.4f}")

        # Save results so far to a FITS file for FittedModel to read
        p_best = {
            "Teff": Teff,
            "logg": logg,
            "rK": rK,
            "vsini": vsini,
            "B": B
        }

        fname = write_grid_to_fits(dirname, i, grid_vals, p_best, chi2, renormalization, shifts,
                        regions, data, resolution, kernel)
        fnames.append(fname)
        print(f"Saved results to {fname}")
        
        # Upsample the grid. Create new parameter grids around the best-fit values
        if i < num_iter - 1:
            grid_vals = upsample_params(grid_vals, p_best, upsamp_factors, steps,
                                        current_iter=i, interpolation_bounds=interpolation_bounds)
            
            print("Requested target grids:")
            for key, val in grid_vals.items():
                print(f"  {key}: {list(val)}")

            # Create upsampled models if they don't exist yet
            for r in regions:
                interpolate_moogstokes_models(
                    models_dir=MoogStokesModel.models_dir,
                    output_dir=MoogStokesModel.models_itp_dir,
                    region=r,
                    target_grids=grid_vals,
                    interpolation_bounds=interpolation_bounds
                )

    return fnames

def min_chi2_slice(chi2_grid, keep_axes):
    # Helper function to compute minimum chi2 slice along all other axes
    all_axes = set(range(chi2_grid.ndim))
    marginal_axes = tuple(all_axes - set(keep_axes))
    return np.min(chi2_grid, axis=marginal_axes)

class FittedModel:
    """
    Class to store target data, chi-squared grid, and model parameters from a
    FITS file generated by the updated grid/upsampling pipeline.
    """

    def __init__(self, datafile: str, interpolation_bounds: dict | None = None,
                 interpolate_chi2: bool = True) -> None:
        hdul = fits.open(datafile)

        hdr = hdul[0].header
        self.name = hdr.get("TARGET")
        self.kernel = hdr.get("KERNEL")
        if self.kernel == 'None':
            self.kernel = None
        self.resolution = hdr.get("RESPWR")
        if self.resolution == -1.0:
            self.resolution = None
        self.data_fname = hdr.get("FNDATA")

        self.chi2_grid = hdul[0].data
        self.renorms = hdul["RENORM"].data
        self.shifts = hdul["SHIFTS"].data
        self.regions = hdul["REGIONS"].data

        # Stored as one ImageHDU per parameter: GRID_<param>
        key_translator = {
            "TEFF": "Teff",
            "LOGG": "logg",
            "RK": "rK",
            "VSINI": "vsini",
            "B": "B"
        }
        self.grid_vals = {}
        for hdu in hdul:
            if hdu.name.startswith("GRID_"):
                key = hdu.name.replace("GRID_", "")
                new_key = key_translator[key]
                self.grid_vals[new_key] = hdu.data.copy()
        self.ordered_params = ["Teff", "logg", "rK", "vsini", "B"]

        # Stored as a single-row binary table
        self.best_params = {}
        pbest_table = hdul["PBEST"].data
        for name in pbest_table.names:
            self.best_params[name] = pbest_table[name][0]
        self.best_params_arr = np.array(
            [self.best_params[key] for key in ["Teff", "logg", "rK", "vsini", "B"]]
        )

        self.param_uncert = {}
        self.chi2_itp = {}
        self._make_uncertainties(interpolate=interpolate_chi2)
        self._make_best_model(interpolation_bounds=interpolation_bounds)

        self.data = SpectralData(self.data_fname, name=self.name)
        yerr_scaling = hdr.get("YERSCL", 1.0)
        self.data.rescale_yerr(yerr_scaling)
        
        hdul.close()

    def _make_uncertainties(self, interpolate=True) -> None:
        """Compute best-fit parameters by interpolating the chi-squared grid.
        """
        Nparams = len(self.best_params)
        delta_chi2_dict = {  # for the number of params, corresponding to 1 sigma confidence
            1: 1.0,
            2: 2.30,
            3: 3.53,
            4: 4.72,
            5: 5.89
        }
        delta_chi2 = delta_chi2_dict[Nparams]

        delta_chi2 = 1

        for i, param in enumerate(self.ordered_params):

            # Margninalize over all other parameters
            grid_vals = self.grid_vals[param]
            chi2_1d = min_chi2_slice(self.chi2_grid, keep_axes=[i])

            if interpolate:
                # Interpolate to find minimum
                itp = interp1d(grid_vals, chi2_1d, kind='cubic')
                fine_grid = np.linspace(grid_vals.min(), grid_vals.max(), 1000)
                chi2_fine = itp(fine_grid)
                self.chi2_itp[param] = (fine_grid, chi2_fine)

            else:
                # Just use the grid
                fine_grid = grid_vals
                chi2_fine = chi2_1d

            chi2_min = np.min(chi2_fine)
            min_idx = np.argmin(chi2_fine)
            best_param = fine_grid[min_idx]
            self.best_params[param] = best_param

            # Estimate 1-sigma uncertainties
            ok = (chi2_fine - chi2_min) <= delta_chi2
            if np.any(ok):
                lo = fine_grid[ok].min()
                hi = fine_grid[ok].max()
            else:
                lo, hi = np.nan, np.nan
            
            self.param_uncert[param] = (best_param - lo, hi - best_param)

    def _make_best_model(self, interpolation_bounds: dict | None = None) -> None:
        """
        Construct a MoogStokesModel for a given region using the best-fit
        parameters stored in this object.

        Parameters
        ----------
        region : int
            Region index to pass to the model.

        Returns
        -------
        model : MoogStokesModel
        """
        print("Checking if best-fit model exists...")
        for r in range(7):
            fname_model = MoogStokesModel.make_model_fname(
                Teff=self.best_params["Teff"],
                logg=self.best_params["logg"],
                rK=self.best_params["rK"],
                B=self.best_params["B"],
                vsini=self.best_params["vsini"],
                region=r,
                models_dir=MoogStokesModel.models_itp_dir,
                use_hash=True
            )
            if not os.path.exists(fname_model):
                # Create interpolated model if it doesn't exist yet
                target_grids = {
                    "Teff": np.array([self.best_params["Teff"]]),
                    "logg": np.array([self.best_params["logg"]]),
                    "rK": np.array([self.best_params["rK"]]),
                    "vsini": np.array([self.best_params["vsini"]]),
                    "B": np.array([self.best_params["B"]])
                }

                print(f"Starting region {r}")
                interpolate_moogstokes_models(
                    models_dir="data/moog-stokes/",
                    output_dir="data/moog-stokes-itp/",
                    region=r,
                    target_grids=target_grids,
                    interpolation_bounds=interpolation_bounds
                )

    def get_best_model(self, region: int, apply_kernel: bool = False) -> MoogStokesModel:
        model = MoogStokesModel(
            Teff=self.best_params["Teff"],
            logg=self.best_params["logg"],
            rK=self.best_params["rK"],
            B=self.best_params["B"],
            vsini=self.best_params["vsini"],
            region=region
        )
        if apply_kernel and (self.resolution is not None or self.kernel is not None):
            model.resolution_change(resolution=self.resolution, Kernel=self.kernel)
        return model

    def plot_data_and_model(self):

        fig, axs = setup_regions_plot()
        obj_test = copy.deepcopy(self.data)

        nn = 0
        for r in range(7):
            ax = axs[r]

            if r in self.regions:
                linecolor = 'fuchsia'
                obj_test.doppler_shift_data(self.shifts[nn])
                obj_test.y *= self.renorms[nn]
                ax.plot(obj_test.x, obj_test.y, c='darkslateblue', zorder=0)
                obj_test.doppler_shift_data(-self.shifts[nn])
                nn += 1

            else:
                linecolor = 'darkred'
                ax.plot(obj_test.x, obj_test.y, c='gray', zorder=0)

            model = self.get_best_model(r)
            ax.plot(model.x, model.y, c=linecolor, alpha=0.7, zorder=1)
            
            ax.text(0.81, 0.05, f"Region {r}", transform=ax.transAxes, color='k')
            ax.set_ylim(min(model.y)-0.05,1.1)

        return fig, axs

    def plot_corner(self, contour_levels=[2.30]):
        """
        Create a corner-like plot of chi-squared from a FittedModel instance.
        Marginalizes all other parameters by taking the minimum chi-squared slice.

        Parameters
        ----------
        fm : FittedModel
            The fitted model containing chi2_grid and parameter grids.
        contour_levels : list[float], optional
            Δχ² levels for 2D contours.
        """
        grids = [self.grid_vals[p] for p in self.ordered_params]
        chi2 = self.chi2_grid
        n = len(self.ordered_params)
        formatted_labels = {
            "Teff": r"$T_{eff}$",
            "logg": r"$\log g$",
            "rK": r"$r_K$",
            "B": r"$B$",
            "vsini": r"$v \sin i$"
        }

        fig, axs = plt.subplots(n, n, figsize=(3*n, 3*n))

        for i in range(n):
            for j in range(n):
                ax = axs[i, j]
                if i == j:
                    param = self.ordered_params[i]

                    # 1D profile along this parameter
                    chi2_1d = min_chi2_slice(chi2, keep_axes=[i])
                    chi2_1d -= np.min(chi2_1d)  # shift so minimum is at 0

                    ax.scatter(grids[i], chi2_1d, color='black', s=10)

                    if param in self.chi2_itp:  # plot the interpolated curve if available
                        xvals = self.chi2_itp[param][0]
                        chi2_1d = self.chi2_itp[param][1]
                        chi2_1d -= np.min(chi2_1d)
                        ax.plot(xvals, chi2_1d, color='gray', alpha=0.5)
                    else:
                        ax.plot(grids[i], chi2_1d, color='gray', alpha=0.5)

                    # Show uncertainties as vertical lines
                    best_val = self.best_params[param]
                    uncert = self.param_uncert[param]
                    ax.axvline(best_val, 0, 100, color='red', linewidth=1)
                    ax.axvline(best_val + uncert[1], 0, 100, color='cornflowerblue', linestyle=':', linewidth=2)
                    ax.axvline(best_val - uncert[0], 0, 100, color='cornflowerblue', linestyle=':', linewidth=2)
                    
                    title = formatted_labels[param] + " = " + f"{best_val:.2f}" + r"$^{+" + f"{uncert[0]:.2f}" + r"}_{-" + f"{uncert[1]:.2f}" + r"}$"
                    ax.set_title(title, fontsize=16)
                    ax.set_xlim(grids[i][0]*0.99, grids[i][-1]*1.01)
                    ax.set_ylim(-0.1, np.max(chi2_1d)*1.05)
                    if i != n - 1:
                        ax.set_xticks([])
                    ax.set_ylabel(r"$\Delta \chi^2$", fontsize=12)
                    ax.yaxis.tick_right()
                    ax.yaxis.set_label_position("right")

                elif i > j:
                    # 2D slice: param i vs param j
                    slice_2d = min_chi2_slice(chi2, keep_axes=[j, i])
                    delta_chi2 = slice_2d - slice_2d.min()
                    delta_chi2 = delta_chi2.T

                    X, Y = np.meshgrid(grids[j], grids[i])
                    ax.imshow(delta_chi2, 
                            origin='lower',
                            aspect='auto',
                            extent=[grids[j][0], grids[j][-1], grids[i][0], grids[i][-1]],
                            cmap="gray")
                    # Show best-fit values as a point
                    best_x = self.best_params[self.ordered_params[j]]
                    best_y = self.best_params[self.ordered_params[i]]
                    ax.scatter(best_x, best_y, color='red', s=20, linewidths=2)
                    ax.vlines(best_x, ymin=grids[i][0], ymax=grids[i][-1], color='red', linewidth=1)
                    ax.hlines(best_y, xmax=grids[j][-1], xmin=grids[j][0], color='red', linewidth=1)

                    # Overlay contours
                    c = ax.contour(X, Y, delta_chi2, levels=contour_levels, colors='cornflowerblue', linestyles=':', linewidths=2)

                    if j == 0:
                        param = self.ordered_params[i]
                        ax.set_ylabel(formatted_labels[param], fontsize=14)
                    else:
                        ax.set_yticks([])
                    if i == n-1:
                        param = self.ordered_params[j]
                        ax.set_xlabel(formatted_labels[param], fontsize=14)
                    else:
                        ax.set_xticks([])
                else:
                    ax.axis("off")
                    
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()

        return fig, axs
