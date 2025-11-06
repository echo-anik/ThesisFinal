# Thesis repository (code-only)

This repository contains the thesis code, preprocessing scripts, experiments and HAI results.

Large datasets are intentionally excluded from this repository. Please upload CSV/raw datasets to your chosen cloud storage (e.g., Google Drive) and place them locally before running preprocessing scripts.

Expected local directories and filenames (place CSV files here):

- processed_data/ (this folder is ignored by git; create locally and place files)
  - wadi_train_raw.csv
  - wadi_test_raw.csv
  - hai_train_raw.csv
  - hai_test_raw.csv

- WADI raw data (optional - place under this folder locally if you keep raw copy):
  - WADI.A2_19 Nov 2019/WADI_14days_new.csv
  - WADI.A2_19 Nov 2019/WADI_attackdataLABLE.csv

How to use
----------
1. Upload the raw CSV files to a local folder `processed_data/` as listed above.
2. Run the preprocessing scripts:
   - `python preprocess_WADI_A2_FINAL.py`
   - `python preprocess_HAI.py`
3. Run experiments:
   - `python run_final_hai_experiments.py`
   - `python run_final_wadi_experiments.py`

Notes
-----
- This repository does NOT include large dataset files to keep the git history small.
- If you need to version large files, consider using Git LFS or external storage.

Contact
-------
If you need help restoring datasets or uploading to GitHub, let me know and I can prepare a bundle or instructions.
