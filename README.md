Detecting Frame Deletion in Videos Using Supervised and Unsupervised Learning with Convolutional Neural Networks
==============================

Table of contents
------------
- [Project Overview](#project-overview)
- [Project Organization](#project-organization)
- [Usage](#usage)
- [Contact](#contact)

Project overview
------------
This is a project for comparing two CNN approaches for detecting frame deletion. While the first one uses 3D CNN, the second one has the capability to adapt multiple CNNs such as VGG-16, Densenet-121, Resnet-50 among others. Both approaches have shown strong performance in terms of accuracy and recall, while also making efficient use of computational resources.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   ├── raw            <- The original, immutable data dump.
    |   └── inference_time <- Sample, to find inference time
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   ├── predict_model.py
        │   └── train_model.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py


Usage
------------
To replicate the experiments from this research, please follow the steps specified in `Docs/getting-started.rst` and also `experiment_1.md` and `experiment_1.md` carefully.

Contact
------------
This article, titled "Detecting Frame Deletion in Videos Using Supervised and Unsupervised Learning with Convolutional Neural Networks", with ID 9568, was developed by researchers from the Universidad San Ignacio de Loyola (USIL):

- **Jorge Ceron** 
Computer Science and Systems Engineer from Universidad San Ignacio de Loyola (USIL), Lima, Peru. He is currently a research member of the Computer Science Research Group (GICC) at USIL. His research interests include machine learning and computer vision. Contact: jorge.ceron@usil.pe

- **Cristian Tinipuclla**
Systems and Informatics Engineer from Universidad San Ignacio de Loyola (USIL), Peru, in 2022. He is currently a research member of the Computer Science Research Group (GICC) at USIL. His research interests include computer vision and neural networks. Contact: cristian.tinipuclla@usil.pe

- **Pedro Shiguihara**
(Senior Member, IEEE) received a master’s degree from the University of S˜ao Paulo, in 2013. He is currently pursuing a Ph.D. degree with the National University of San Marcos. He heads the Computer Science Research Group of the Universidad San Ignacio de Loyola, Lima, Peru. Contact: pshiguihara@usil.pe

<p><small>This project was implemented with <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
