"""Immutable physical constants (CGS) for stellar-jax.

These are fundamental physics constants that never change. No calibration
results, no solver tuning, no mesh parameters — those live elsewhere in
this package.

References:
- G: MESA const_def.f90:114 standard_cgrav = 6.67430d-8
- mu_sun: IAU 2015 Res B3 solar mass parameter (1.3271244e26 cm^3 s^-2)
- Msun: derived mu_sun/G (matching MESA const_def.f90:124)
- Lsun: IAU 2015 Res B3 (3.828e33 erg/s), MESA const_def.f90:126
- Rsun: IAU 2015 Res B3 (6.957e10 cm), MESA const_def.f90:125
- c_light: exact (2.99792458e10 cm/s)
- sigma_sb: Stefan-Boltzmann (5.67051e-5 erg cm^-2 s^-1 K^-4)
- a_rad: radiation constant (4*sigma_sb/c)
- k_B: Boltzmann (1.380658e-16 erg/K)
- m_H: hydrogen mass (1.673534e-24 g)
- Y_BBN: Planck 2018 + BBN (Pitrou et al. 2018)
- DY_DZ: He enrichment law slope (Balser 2006, AJ 132, 2326)
"""

# Fundamental constants (CGS)
G = 6.67430e-8               # gravitational constant [dyn cm^2 g^-2]
                             # MESA const_def.f90:114 standard_cgrav
e_cgs = 4.80320451e-10       # electron charge [esu / statcoulomb]
c_light = 2.99792458e10      # speed of light [cm/s]
sigma_sb = 5.67051e-5        # Stefan-Boltzmann [erg cm^-2 s^-1 K^-4]
a_rad = 7.56591e-15          # radiation constant [erg cm^-3 K^-4]
k_B = 1.380658e-16           # Boltzmann constant [erg/K]
m_H = 1.673534e-24           # hydrogen atom mass [g]

# Solar values — MESA const_def.f90:118-126, IAU 2015 Resolution B3
mu_sun = 1.3271244e26        # solar mass parameter [cm^3 s^-2] (IAU 2015 Res B3)
Msun = mu_sun / G            # solar mass [g] — derived, matching MESA const_def.f90:124
Lsun = 3.828e33              # solar luminosity [erg/s] (IAU 2015 Res B3)
Rsun = 6.957e10              # solar radius [cm] (IAU 2015 Res B3)
T_sun = 5778.0               # nominal solar effective temperature [K]
                             # (pre-IAU convention; Mamajek et al. 2015, arXiv:1510.07674)
log_g_sun = 4.438            # log10(g_sun / cm s^-2), g_sun = G*Msun/Rsun^2
                             # (Prša et al. 2016, AJ 152, 41; IAU 2015 Res B3 derived)

# Cosmological / initial composition
Y_BBN = 0.2485               # primordial He mass fraction (Planck 2018 + BBN)
DY_DZ = 1.5                  # He enrichment law slope (Balser 2006)
SECONDS_PER_YEAR = 3.15576e7 # Julian year [s]

# Nuclear energy release
Q_PER_G = 0.007 * c_light**2  # energy per gram of H burned (pp-chain) [erg/g]

# --- Microphysics module constants (exact values used by HELM/FD EOS) ---
# These differ in precision from the "primary" constants above. They exist to
# preserve bit-identical behavior across the codebase — each module uses the
# exact value it was originally written with (Timmes & Swesty 2000 precision).
#
# HELM EOS (Timmes & Swesty 2000; Chabrier & Potekhin 1998):
k_B_helm = 1.380649e-16       # Boltzmann constant [erg/K] — CODATA 2018 exact
m_H_helm = 1.6726e-24         # proton mass [g]
m_e_helm = 9.1094e-28         # electron mass [g]
hbar_helm = 1.0546e-27        # reduced Planck constant [erg s]
c_light_helm = 2.9979e10      # speed of light [cm/s]
e_charge_helm = 4.8032e-10    # electron charge [esu]
a_rad_helm = 7.5657e-15       # radiation constant [erg cm^-3 K^-4]
#
# FD electron EOS (Timmes & Swesty 2000; Aparicio 1998):
k_B_fd = 1.380649e-16         # Boltzmann constant [erg/K] — same as HELM
m_H_fd = 1.6726219237e-24     # proton mass [g] — higher precision
m_e_fd = 9.1093837015e-28     # electron mass [g] — higher precision
hbar_fd = 1.054571817e-27     # reduced Planck constant [erg s]
c_light_fd = 2.99792458e10    # speed of light [cm/s] — exact
e_charge_fd = 4.8032e-10      # electron charge [esu]
a_rad_fd = 7.5657e-15         # radiation constant [erg cm^-3 K^-4]
#
# OPAL EOS internal (lower precision — used in eos.py functions):
k_B_opal = 1.3806e-16         # Boltzmann constant [erg/K]
m_H_opal = 1.6726e-24         # proton mass [g]
h_planck_opal = 6.6261e-27    # Planck constant [erg s]
a_rad_opal = 7.5657e-15       # radiation constant [erg cm^-3 K^-4]
