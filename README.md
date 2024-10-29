GMV with Foreground Mitigation
====

These scripts are meant for use with the [GMV curved-sky pipeline](https://github.com/yomori/healqest).

The most updated scripts are in the `new` directory.

----------

First, get the plm reconstruction results using `get_plms_standard.py` (similarly, use `get_plms_mh_crossilc.py` for MH or cross-ILC). The script can be run for one realization like
```
python3 get_plms_standard.py TT 100 100 standard_cinv test_yuka.yaml
```
where the first argument is the estimator (in the case of MH and cross-ILC, we want `T1T2` or `T2T1` instead of `TT`), the second two arguments are the sim numbers for the input maps, the next argument can be e.g. `standard`, `standard_cinv`, `standard_cmbonly_phi1_tqu1tqu2`, etc. (see comments at the top of the `get_plms_standard.py` file for a better description of what this argument should be, and also see `slurm_get_plms_standard.sh` for thorough examples), and the last argument is the path to the config file.

The paths to the input maps, ILC weights, output directory, etc. are all hard-coded into `get_plms_standard.py`, so make sure you change all of that before running (and please don't overwrite my test files)!

I am using 250 total sims for my tests, and baselining cinv-style. To get all the plms required for a full analysis of the standard case (cinv-style), I used `slurm_get_plms_standard.sh`. See this submit script for a good example of how `get_plms_standard.py` should be run. It takes around 3 minutes for `get_plms_standard.py` to get the plm for one estimator for one pair of input sims.

Note: Before `get_plms_standard.py` can actually output plms, you need the averaged total Cls for this sim set. See lines 370 and on. If you are using the same input maps as me, you can just use the total Cls I have saved on the Sherlock cluster (path in line 374 of the `get_plms_standard.py` script). If not, you need to save the total Cls for each input sim (`get_plms_standard.py` will do this automatically if it can't find a saved averaged totalcls file, see lines 379 and on) and then manually average them using something like `average_totalcls.py` before proceeding.

Another note: For MH and cross-ILC, in `get_plms_mh_crossilc.py`, there is an additional argument that should either be `withT3` or `noT3`. The `withT3` case means you're always using the MV-ILC "T3" map for estimators like TE, ET, TB, BT. The `noT3` case means you're using T1 (which is the MV-ILC T map for MH and CIB-nulled T map for cross-ILC) for TE, TB and using T2 (tSZ-nulled T map) for ET, BT, without using T3 (filtering is adjusted accordingly). For examples of usage, see `slurm_get_plms_mh_crossilc.sh`. For my tests, I am baselining `noT3` because I found that to be slightly less noisy than `withT3` (which is not intuitive, and I need to check if that is actually right, but that is a whole other rabbit hole...).

Once you have all the necessary plms, use `analyze_standard.py` to do the sim response, N0, and N1 computations (similarly `analyze_mh_crossilc.py` for foreground mitigation cases). With 250 sims, it takes about an hour for me to loop through and get the N0, and two hours for N1 (since remember, we need to get the CMB-only N0 as part of the N1 calculation).

Once you have the response, N0, and N1 from sims, you can use the Agora realizations as your "real data" to get the lensing bias. Use `slurm_get_plms_agora.sh` to call `get_plms_agora.py` to get the Agora plms. Then, `analyze_agora.py` does the actual lensing bias calculations and plots. The RDN0 computation is also done here.

I know the code is very long and hard to parse in some spots even though I tried my best to put in comments. Please contact me if you have any questions or want me to retrieve any files or anything like that. You can reach me on Slack or email (yukanaka@stanford.edu) during my deployment (Nov 2024 to Feb 2025). The satellite schedule for South Pole Station is updated weekly and posted [here](https://www.usap.gov/technology/1935/) if you want to see the times when I'll have Internet.
