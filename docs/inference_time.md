# Introduction

- Este proceso será usado para medir el tiempo de inferencia de los modelos del trabajo.
- Por favor seguir los siguientes pasos:

## General
1. Activar environment `main_env` el cual se obtiene al instalar main_env_v3.yml, ubicado en {raiz_proyecto}/docs/main_env_v3.yml. Para instalar ubicarse en la raiz del proyecto y ejecutar: `conda env create -f docs/main_env_v3.yml`
2. En la raíz del proyecto crear la carpeta "data", según el siguiente organigrama:
------------
    ├── Raíz del proyecto
        └── data
-----------
3. Ubicarse dentro de la carpeta "data", y ejecutar los siguiente comandos en orden:
- `wget "https://usilpe-my.sharepoint.com/:u:/g/personal/jorge_ceron_usil_pe/ETy9sYDZyxJLtXKe4RFLg1cB6Gqiol6j8DbXdnXShMyvtw?e=36ucUv&download=1" -O inference_time.zip`
- `unzip inference_time.zip`


## Prototipo 1
1. Posicionarse en la raiz del proyecto
2. Ejecutar:
    - `python src/features/inference_time.py -m "model1" -vp data/inference_time --cuda`

## Prototipo 2 - Original
1. Ejecutar:
    - `python src/features/inference_time.py -m "model2_original" -vp data/inference_time --cuda`

## Prototipo 2 - VGG16
1. Ejecutar:
    - `python src/features/inference_time.py -m "model2_vgg16" -vp data/inference_time --cuda`

## Prototipo 2 - Resnet50
1. Ejecutar:
    - `python src/features/inference_time.py -m "model2_resnet" -vp data/inference_time --cuda`

## Prototipo 2 - Nasnet Large
1. Ejecutar:
    - `python src/features/inference_time.py -m "model2_nasnet" -vp data/inference_time --cuda`

## Prototipo 2 - Inception V3
1. Ejecutar:
    - `python src/features/inference_time.py -m "model2_inceptionv3" -vp data/inference_time --cuda`

## Prototipo 2 - DenseNet
1. Ejecutar:
    - `python src/features/inference_time.py -m "model2_densenet" -vp data/inference_time --cuda`
  
## Paso final
Subir a Gdrive el archivo csv contenido en la {raiz_proyecto}/reports/inference_time/inference_time.csv
