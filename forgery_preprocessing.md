# Introduction
- Purpose: Remove a certain number of frames per video.

## Environment
1. Create a conda environment: `conda create -n forgery_env python=3.10` and activate it.
2. `pip install opencv-python`.
3. `conda install anaconda::pandas`.

## Execution
1. In a dataset (D1, D2, etc.), create the following folders: 
  - *data*: Folder with original videos from the dataset.
  - *forgery_data*: Manipulated and original videos.
  - *forgery_labels*: .npy files for labeling.
  - *forgery_info*: Information about manipulations in a .csv file.
  - *p1_data_batches*: Videos in .npy batches.
  - *p1_label_batches*: Batch labels in .csv.
2. Forgery task: Activate the `forgery_env` environment and run `python forgery_module/main.py -d D1/data/ -f D1/_module_data/ -l D1/forgery_labels -fd D1/forgery_info/`.
3. Preprocessing task: Run `python prototype_1/preprocessing.py -fd D1/forgery_data/ -fl D1/forgery_labels/ -db D1/p1_data_batches/ -lb D1/p1_label_batches/`.
