# =========================================================
# Import all dependencies
# =========================================================
import numpy as np
#
# Define the ZTF bands in Angstroms (ZTF mean filter wavelengths)
#
ztf_band = {'g': np.log10(4746.48), 'r': np.log10(6366.38), 'i': np.log10(7829.03)}
#
# The ZTF mag saturation and limit values are found based on 99% of the standardized magnitude of the largest dataset
#
ztf_mag = {'saturation':-0.6958509378753988, 'limit':1.5205410620126045}