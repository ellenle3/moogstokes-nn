from astropy.convolution import Gaussian1DKernel, convolve, Box1DKernel
import numpy as np
from spectra import *
import matplotlib.pyplot as plt


def Lorentz_Cauchy(sigma, x):
    x0 = x[len(x) // 2]  # center on the middle of the array
    kernel = (1 / np.pi) * sigma / ((x - x0)**2 + sigma**2)
    return kernel / np.sum(kernel)  # normalize

def instrumental_response(flux, wavelength, resolution, Kernel='box', reference_wavelength = None):
    """
    Instrumental response of the spectrograph convolution. Two convolution kernels
    are available for convolution, Gaussian and Boxcar.
    INPUT
        -Model_flux: the data to convolve
        -resolution: Spectral resolution of the spectrograph
        in Angstroms*Npoints, Npoints the number of spectral
        points per angstrom. E.g if the model was sampled as 0.1 Angstrom (10 points per Angstrom)
        and we want R=2000 at 2.0 microns. resolution=100 (10*10)
        -Kernel: Either box or gaussian
         BOX kernel uses directly the resolution element
         Gaussian uses the FWHM, so 68 percent of the width
    OUTPUT
    """
    #### spacing in wavelength of the data or model to be convolved

    dx = abs(wavelength[1]-wavelength[0])

    # print('wavelength spacing of the model/data is ',dx, 'pixels per wavelength unit, .e.g., angstroms')
    length_model = len(flux)

    if Kernel == 'box':
        kernel = Box1DKernel(resolution, mode='linear_interp')

    elif Kernel == 'SPEX':
        if reference_wavelength is None:
            reference_wavelength = np.nanmean(wavelength)

        sigma_wavelength_unit = reference_wavelength/resolution / (2 * np.sqrt(2 * np.log(2))) ## sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
        sigma_pixels = sigma_wavelength_unit / dx

        x_size = int(8 * sigma_pixels) | 1  # bitwise OR with 1 -> force odd
        x_size = min(x_size, length_model)  # don't exceed data length

        kernel = Gaussian1DKernel(sigma_pixels, x_size=x_size)

        # print(sigma_pixels)
        # plt.plot(kernel_not_normalized, drawstyle='steps')
        # plt.show()

    elif Kernel == 'KECK':
        """
        Convolve the data with the analytical form obtained for iSHELL K2 0.375''
        """

        L_fwhm=0.043
        G_fwhm=0.446
        B_fwhm=0.649

        pixel_spacing=resolution

        sigma_g = (G_fwhm/pixel_spacing) / (2 * np.sqrt(2 * np.log(2)))
        kernel_g = Gaussian1DKernel(sigma_g)

        if len(flux) % 2 == 0:
            x = np.linspace(0, len(flux), len(flux) - 1)
        else:
            x = np.linspace(0, len(flux), len(flux))
        sigma_l = L_fwhm/pixel_spacing
        kernel_l = Lorentz_Cauchy(sigma_l, x)

        kernel_voigt = convolve(kernel_l, kernel_g, boundary='extend')
        kernel_voigt_norm = kernel_voigt / sum(kernel_voigt)

        Box = Box1DKernel(B_fwhm/pixel_spacing, mode='linear_interp')

        kernel_not_normalized = convolve(kernel_voigt_norm, Box, boundary='extend')

        kernel = kernel_not_normalized/ sum(kernel_not_normalized)

        # plt.plot(kernel,marker='.')
        # plt.show()

    # kernel = kernel_not_normalized/ sum(kernel_not_normalized)
    # kernel = kernel_not_normalized


    Convolved_data = convolve(flux, kernel, boundary='extend')

    return Convolved_data



if __name__ == "__main__":

    ### Test if everything is ok
    fname = "data/science/Spectrum_IRAS03301+3111.nspec"

    # ProplydData is a simple class that stores the data and has some built-in methods
    # to help with data processing
    obj = SpectralData(fname)

    # Let's nyquist sample the data. This assumes that we are using iSHELL 0.75'' slit width with is
    # over sampled by 6 pixels per resolution element. We are bring it down to 2 (Nyquist)
    # obj.Nyquist_bin_spectrum(N = 3)

    obj.x *= 1  # e4  # convert x-axis to angstroms
    # obj.x *= 1e4  # convert x-axis to angstroms
    print(type(obj.x))

    new_flux = instrumental_response(flux=obj.y, wavelength=obj.x, resolution=3000, Kernel='SPEX', reference_wavelength=22500)

    plt.figure()
    plt.plot(obj.x, obj.y)
    plt.plot(obj.x,new_flux)
    plt.ylim(0.7, 1.1)
    plt.xlim(22600, 22680)
    plt.title(obj.name)  # the object name was automatically set from the file name - see ProplyData.__init__()
    plt.show()
