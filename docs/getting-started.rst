Getting started
===============

.. This is where you describe how to get set up on a clean install, including the
   commands necessary to get the raw data (using the `sync_data_from_s3` command,
   for example), and then how to make the cleaned, final data sets.

Creación de environments
------------------------

Environment para prototipos
^^^^^^^^^^^^^^^^^^^^^^^^^^^
- El environment, entre sus dependencias más importantes, incluye: Python 3.10, Opencv, Tensorflow 2.15 (and cuda), scikit learn, Matplotlib (P2), Scipy (P2) and Pandas.

1. Posicionarse en raíz de proyecto.
2. Ejecutar: `conda env create -f docs/main_env_v2.yml`.  

Environment para demo jupyter notebook
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. `conda create -n demo_env python`.
2. `conda install nb_conda`.
3. `conda install anaconda::ipywidgets`.
4. `conda install conda-forge::opencv`.
5. `conda install anaconda::numpy`.
6. `conda install conda-forge::matplotlib`.
7. `pip install tensorflow-cpu`.
8. `conda install scipy`.


Forgery Processing 
------------------

- Propósito: Eliminar cierta cantidad de frames por video.

Pasos
^^^^^
1. Utilizar el environment para prototipos.
2. En un dataset (D1, D2, etc), crear las siguientes carpetas: 
  - *data*: Carpeta con videos originales de dataset.
  - *forgery_data*: Videos manipulados y originales.
  - *forgery_labels*: Archivos .npy para etiquetado.
  - *forgery_info*: Información de manipulaciones en un archivo .csv.
  - *p1_data_batches*: Videos en batches .npy.
  - *p1_label_batches*: Etiquetas de batches en .csv
3. Forgery task: Activar environment `forgery_env` y ejecutar `python forgery/main.py -d D1/data/ -f D1/forgery_data/ -l D1/forgery_labels -fd D1/forgery_info/`.
4. Preprocessing task para P1: Ejecutar `python prototype_1/preprocessing.py -fd D1/forgery_data/ -fl D1/forgery_labels/ -db D1/p1_data_batches/ -lb D1/p1_label_batches/`.
5. Preprocessing task para P3: Ejecutar `python ./src/features/preprocessing/model_3_ours_preprocess.py -fd ./data/D3/forgery_data -fl ./data/D3/forgery_labels -db ./data/D3/p3_data_batches/ -lb ./data/D3/p3_label_batches/`