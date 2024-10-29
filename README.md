GMV with Foreground Mitigation
====

These scripts are meant for use with the [GMV curved-sky pipeline](https://github.com/yomori/healqest).

The most updated scripts are in the `new` directory.

----------

First, get the plm reconstruction using `get_plms_standard.py` (similarly, use `get_plms_mh_crossilc.py` for MH or cross-ILC). The script can be run for one realization like
```
python3 get_plms_standard.py TT 100 100 standard_cinv test_yuka.yaml
```
where the first argument is the estimator (in the case of MH and cross-ILC, we want `T1T2` or `T2T1` instead of `TT`), the second two arguments are the sim numbers for the input maps, the next argument can be e.g. `standard`, `standard_cinv`, `standard_cmbonly_phi1_tqu1tqu2`, etc. (see comments at the top of the `get_plms_standard.py` file for a better description of what this argument should be), and the last argument is the path to the config file.

The paths to the input maps, ILC weights, output directory, etc. are all hard-coded into `get_plms_standard.py`, so make sure you change all of that before running!

I am using 250 total sims for my tests. To get all the plms required for a full analysis of the standard case (cinv-method), I used `slurm_get_plms_standard.sh`. See this submit script for a good example of how `get_plms_standard.py` should be run.

Note: Before `get_plms_standard.py` can actually be run, you need the averaged total Cls for this sim set. See lines 370 and on. If you are using the same input maps as me, you can just use the total Cls I have saved on the Sherlock cluster (path in line 374 of the `get_plms_standard.py` script). If not, you need to save the total Cls for each input sim (`get_plms_standard.py` will do this automatically if it can't find a saved averaged totalcls file) and then manually average them using something like `average_totalcls.py` before proceeding.

Once you have all the necessary plms, use `analyze_standard.py` to do the sim response, N0, and N1 computations.
