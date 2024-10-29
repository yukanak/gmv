GMV with Foreground Mitigation
====

These scripts are meant for use with the GMV curved-sky pipeline at `https://github.com/yomori/healqest`.

The most recent scripts are in the `new` directory.

First, get the plm reconstruction using `get_plms_standard.py` (similarly, use `get_plms_mh_crossilc.py` for MH or cross-ILC). The script can be run like
```
python3 get_plms_standard.py TT 100 101 append test_yuka.yaml
```
where the first argument is the estimator (in the case of cross-ILC, it wants arguments `T1T2` or `T2T1` instead of `TT`), the second two arguments are the sim numbers, `append` can be e.g. `standard`, `standard_cinv`, `standard_cmbonly_phi1_tqu1tqu2`, etc. (see comments at the top of the file for a better description of what the `append` argument should be), and the last argument is the path to the config file containing information for the lmax, etc.
