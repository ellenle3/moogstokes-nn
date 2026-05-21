"""
From a private communication with Christian. I modified it a bit so it will
take iSHELL FITS file and the BT-Settl templates. Wavelength units are all
converted to Angstroms. The BT-Settl spectra are also in air wavelengths by
default. If a template spectrum is loaded, the wavelengths will be shifted so 
the saved *.nspec spectrum is in terms of vacuum wavelengths.

This is a nice interactive continuum normalization routine that Christian found
in Python4Astronomers:
ftp://ftp.ster.kuleuven.be/dist/pierre/Mike/IvSPythonDoc/plotting/specnorm.html

Usage:
1-Click with the left button on the continuum part of the spectrum,
2-when enough points were selected, press the "enter" key, to fit
a polinomial
3-if you are happy with the polinomial press the "n" key to normalize
4-To write the normalize spectrum into a data file, press "w"
---
5- Click with the right button to un-select points it the spectrum
6- Press "r" key at any time to reset to the original spectrum
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splrep, splev
import sys
import os
import copy

from spectra import *


BAND_W_LIMS = {'J': [12600, 13700], 'K': [20800, 24000], 'M': [45100, 52300]}

# Mutable state dict — avoids global/local scoping issues in event handlers
state = {}


def onclick(event):
    toolbar = plt.get_current_fig_manager().toolbar
    if (event.button == 1
            and toolbar.mode == ''
            and not event.dblclick
            and event.xdata is not None
            and event.ydata is not None):
        plt.plot(event.xdata, event.ydata, 'gs', ms=5, picker=2, label='cont_pnt')
        plt.draw()


def onpick(event):
    if event.mouseevent.button == 3:
        if hasattr(event.artist, 'get_label') and event.artist.get_label() == 'cont_pnt':
            event.artist.remove()
    plt.draw()


def ontype(event):
    wave     = state['wave']       # unbinned
    flux     = state['flux']       # unbinned
    uncert   = state['uncert']     # unbinned
    filename = state['filename']
    data_binned = state['data_binned']

    if event.key == 'enter':
        cont_pnt_coord = []
        for artist in plt.gca().get_children():
            if hasattr(artist, 'get_label') and artist.get_label() == 'cont_pnt':
                cont_pnt_coord.append(artist.get_data())
            elif hasattr(artist, 'get_label') and artist.get_label() == 'continuum':
                artist.remove()
        cont_pnt_coord = np.array(cont_pnt_coord)[..., 0]
        sort_array = np.argsort(cont_pnt_coord[:, 0])
        x, y = cont_pnt_coord[sort_array].T
        spline = splrep(x, y, k=1)
        continuum = splev(wave, spline)   # evaluated at unbinned wavelengths
        plt.plot(wave, continuum, 'r-', lw=2, label='continuum')

    elif event.key == 'n':
        continuum = None
        for artist in plt.gca().get_children():
            if hasattr(artist, 'get_label') and artist.get_label() == 'continuum':
                continuum = artist.get_data()[1]
                break
        if continuum is not None:
            norm_flux = flux / continuum
            norm_err  = uncert / continuum
            plt.cla()
            plt.plot(wave, norm_flux, 'k-', label='normalised')
            plt.plot(wave, norm_err,  'r-', label='uncertainty')
            finite = norm_flux[np.isfinite(norm_flux)]
            if finite.size > 0:
                ymed = np.median(finite)
                plt.ylim(ymed / 2, ymed * 1.5)
            plt.plot(wave, np.ones_like(wave), c='cyan', alpha=0.5)

    elif event.key == 'u':
        cont_points = [a for a in plt.gca().get_children()
                       if hasattr(a, 'get_label') and a.get_label() == 'cont_pnt']
        if cont_points:
            cont_points[-1].remove()

    elif event.key == 'r':
        plt.cla()
        plt.plot(data_binned.x, data_binned.y, 'k-', linewidth=0.7)
        ymed = np.nanmedian(flux)
        plt.ylim(ymed / 2, ymed * 1.5)

    elif event.key == 'w':
        normalized_flux = []
        normalized_err  = []
        for artist in plt.gca().get_children():
            if hasattr(artist, 'get_label') and artist.get_label() == 'normalised':
                normalized_flux = np.array(artist.get_data())[1]
                print('flux saved to file')
            elif hasattr(artist, 'get_label') and artist.get_label() == 'uncertainty':
                normalized_err = np.array(artist.get_data())[1]
                print('error saved to file')
                break

        with open(os.path.splitext(filename)[0] + '.nspec', 'w') as f:
            for ii in range(len(wave)):
                f.write('{:15} {:>20} {:>20}\n'.format(
                    wave[ii] * 1e4, normalized_flux[ii], normalized_err[ii]))
        sys.exit()

    plt.draw()


if __name__ == "__main__":
    filename = sys.argv[1]
    data = SpectralData(filename)

    # Copy unbinned arrays before any modification
    state['wave']     = np.array(data.x)
    state['flux']     = np.array(data.y)
    state['uncert']   = np.array(data.yerr)
    state['filename'] = filename

    # Binned copy for display only
    data_binned = copy.deepcopy(data)
    data_binned.Nyquist_bin_spectrum(3)
    state['data_binned'] = data_binned

    plt.plot(data_binned.x, data_binned.y, 'k-', label='spectrum', linewidth=0.7)
    ymed = np.nanmedian(state['flux'])
    plt.ylim(ymed / 2, ymed * 1.5)
    plt.title(filename)

    plt.gcf().canvas.mpl_connect('button_press_event', onclick)
    plt.gcf().canvas.mpl_connect('pick_event',         onpick)
    plt.gcf().canvas.mpl_connect('key_press_event',    ontype)
    plt.show()
