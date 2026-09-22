"""MLT geometry constants — Henyey option (Cox & Giuli 1968 parameterization).

Matches MESA turb/private/mlt.f90 lines 90–95 (case 'Henyey'):
  ff1 = 1.0d0/Henyey_MLT_nu_param   (nu_param = 8.0)
  ff2 = 0.5d0
  ff3 = 8.0d0/Henyey_MLT_y_param    (y_param = 1/3)
  ff4 = 1.0d0/Henyey_MLT_y_param    (y_param = 1/3)
"""

NU_PARAM = 8.0
Y_PARAM = 1.0 / 3.0

FF1 = 1.0 / NU_PARAM    # = 0.125
FF2 = 0.5
FF3 = 8.0 / Y_PARAM     # = 24.0
FF4 = 1.0 / Y_PARAM     # = 3.0
