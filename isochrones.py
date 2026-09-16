import os
import glob
import numpy as np
from scipy.interpolate import LinearNDInterpolator, CubicSpline

class EvolutionaryModelInterpolator:
    """Get mass (Msun), age (Myr) for an input Teff, logg. For example:

    >>> evitp = EvolutionaryModelInterpolator()
    >>> EvolutionaryModelInterpolator(3600, 3.5)
    (array([0.5109306]), array([1.17708182]))

    Can also pass arrays as inputs.

    Attributes
    ----------
    isochrones: (ages, iso_all)
        See import_evolutionary_models()
    
    mass_tracks: (masses, mass_all)
    """

    @staticmethod
    def parse_isochrone(fname: str) -> tuple[list, list, list]:
        arr = np.loadtxt(fname, skiprows=5).T
        m = arr[0]
        teff = 10**arr[1]      # convert from logTeff to Teff
        logg = arr[2]
        return m, teff, logg

    @staticmethod
    def parse_mass_track(fname) -> tuple[list, list, list]:
        arr = np.loadtxt(fname, skiprows=4, usecols=(0, 1, 2)).T
        age = arr[0]
        teff = 10**arr[1]      # convert from logTeff to Teff
        logg = arr[2]
        return age, teff, logg

    def __init__(self,
                iso_path: str = "data/isochrones/magnetic_cleaned",
                mass_path: str = "data/mass-tracks/magnetic_cleaned",
                iso_pattern: str = "dmestar_*myr_z+0.00_a+0.00_gs98_phx_magBeq.iso",
                mass_pattern: str = "m*_GS98_p000_p0_y28_mlt1.884_magBeq.trk") -> None:
        
        self.iso_path = iso_path
        self.mass_path = mass_path
        self.iso_pattern = iso_pattern
        self.mass_pattern = mass_pattern

        self.import_evolutionary_models()
        self.make_evolutionary_interpolators()

    def import_evolutionary_models(self) -> None:
        """Import Feiden (2016) evolutionary models.

        Returns
        -------
        ages: list of size n
            Age of each isochrones.
        iso_all: list of size (n,)
            Isochrones for each age. Each isochrone contains: [ [masses], [Teffs], [loggs] ]
        masses, mass_all:
            Same as above but for mass tracks. The first list in each track is the age.
        """

        fnames_iso = glob.glob( os.path.join(self.iso_path, self.iso_pattern) )
        fnames_iso.sort()
        # remove ages older than 10 Myr
        fnames_iso = [f for f in fnames_iso if float(os.path.basename(f).split("_")[1][:-3]) <= 10]

        # remove masses greater than 1 Msun
        fnames_mass = glob.glob( os.path.join(self.mass_path, self.mass_pattern) )
        fnames_mass.sort()
        fnames_mass = [f for f in fnames_mass if int(os.path.basename(f).split("_")[0][1:]) <= 1000]

        iso_all = []
        ages = []
        mass_all = []
        masses = []

        for fname in fnames_iso:
            mass, teff, logg = self.parse_isochrone(fname)
            iso_all.append( (mass, teff, logg) )
            ages.append( float(os.path.basename(fname).split("_")[1][:-3]) )

        for fname in fnames_mass:
            age, teff, logg = self.parse_mass_track(fname)
            # filter out ages
            mask = age >= 0.07e6
            age = age[mask]
            teff = teff[mask]
            logg = logg[mask]
            mass_all.append( (age, teff, logg) )
            masses.append( int(os.path.basename(fname).split("_")[0][1:]) / 1000.0 )

        self.isochrones = (ages, iso_all)
        self.mass_tracks = (masses, mass_all)

    def make_evolutionary_interpolators(self) -> None:

        ages, iso_all = self.isochrones
        masses, mass_all = self.mass_tracks
        
        # store regions of logg, Teff that are invalid
        junk, teff, logg = iso_all[0]
        self.min_age_itp = CubicSpline(teff, logg)

        junk, teff, logg = mass_all[0]
        self.min_mass_itp = CubicSpline(logg, teff)

        # make interpolators - use log of Teff to make values more well-behaved
        iso_logteffs = np.log10(np.concatenate([iso[1] for iso in iso_all]))
        iso_loggs    = np.concatenate([iso[2] for iso in iso_all])
        iso_ages     = np.concatenate([np.full_like(iso[1], ages[i]) for i, iso in enumerate(iso_all)])

        points = np.column_stack([iso_logteffs, iso_loggs])
        self.age_itp  = LinearNDInterpolator(points, iso_ages)

        trk_logteffs = np.log10(np.concatenate([trk[1] for trk in mass_all]))
        trk_loggs    = np.concatenate([trk[2] for trk in mass_all])
        trk_masses   = np.concatenate([np.full_like(trk[1], masses[i]) for i, trk in enumerate(mass_all)])

        points = np.column_stack([trk_logteffs, trk_loggs])
        self.mass_itp = LinearNDInterpolator(points, trk_masses)

    def __call__(self, Teff: float | NDArray, logg: float | NDArray
                 ) -> tuple[NDArray, NDArray]:

        Teff = np.atleast_1d(Teff).astype(float)
        logg = np.atleast_1d(logg).astype(float)

        q = np.column_stack([np.log10(Teff), logg])  # log10 to match interpolator
        mass, age = self.mass_itp(q), self.age_itp(q)

        # Mask out invalid regions
        Npts = len(Teff)
        for i in range(Npts):
            Teff_in = Teff[i]
            logg_in = logg[i]

            # Above the minimum age line
            if logg_in < self.min_age_itp(Teff_in):
                mass[i] = np.nan
                age[i] = np.nan

            # To the right of the minimum mass line
            if Teff_in < self.min_mass_itp(logg_in):
                mass[i] = np.nan
                age[i] = np.nan

        return mass, age