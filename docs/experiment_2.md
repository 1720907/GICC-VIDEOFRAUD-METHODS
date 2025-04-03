# Introduction

- Este experimento consiste en aplicar 5-fold-cross-validation con P1 y P2 en los conjuntos de datos D1, D2 y D3 (y sus combinaciones). Las salidas serán ypred e ylabels para cada caso, para finalmente pasarlas por un algoritmo comparativo.

## Dataset

1. Git clone o git pull: `git clone https://github.com/gicclab/gicc-scrum_videofraud_metodos`.
2. Crear la siguiente carpeta en la raiz del proyecto:
-----------
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
-----------
3. Descargar siguientes datasets:
  - D1: `wget "https://usilpe-my.sharepoint.com/:u:/g/personal/jorge_ceron_usil_pe/EUu7H5KNsB9OtOBPODQjl0wB6cZfGYbuPR2zYmU3SyXeyA?e=GCOMwp&download=1" -O D1.zip`
  - D2: `wget "https://usilpe-my.sharepoint.com/:u:/g/personal/jorge_ceron_usil_pe/EaAcV5AGQctCm4MmsawQ9WIB7bMYrDIvfQOgiLXpsGYWVQ?e=L6RDP5&download=1" -O D2.zip`
  - D3: `wget "https://usilpe-my.sharepoint.com/:u:/g/personal/jorge_ceron_usil_pe/EYh-I_CR4C1PsNlvWc5PkScBBpl1tj4PfyyOdWGGhKp0dA?e=DOgLGa&download=1" -O D3.zip`
4. Descromprimirlos en `data/processed`
5. Ubicados en la carpeta `data/processed`, ejecutar `cp ../../docs/main.sh main.sh`, y luego ejecutar `bash main.sh` para que se creen las carpetas de los datasets restantes (D1+D2, D2+D3, D1+D3, D1+D2+D3).
6. Ubicados en la raiz del proyecto descargar el preprocessing para P3 con el comando: `wget "https://usilpe-my.sharepoint.com/:u:/g/personal/jorge_ceron_usil_pe/EQnGpAJqEzJFk7MVnoTnPpkB2ciWf3Ud5cQoDazEkSwQxQ?e=TUTsZ9&download=1" -O prepro_P3.zip`
7. Ejecutar el comando `unzip prepro_P3.zip`

## Prototipo 1

1. Instalar environment para P1 (`main_env_v2.yml`) siguiendo las instrucciones de environment para prototipos de `getting-started.rst`.
2. Activar environment `main_env`.
3. Ejecutar 5 fold cross validation para P1(windows):
      - `python .\src\models\model_1\train_test_crossed_fold.py .\data\processed\D1 -ch .\data\processed\D1\p1_output -e 100 -f 5 --cuda`.
      - `python .\src\models\model_1\train_test_crossed_fold.py .\data\processed\D2 -ch .\data\processed\D2\p1_output -e 100 -f 5 --cuda`.
      - `python .\src\models\model_1\train_test_crossed_fold.py .\data\processed\D3 -ch .\data\processed\D3\p1_output -e 100 -f 5 --cuda`.
      - `python .\src\models\model_1\train_test_crossed_fold.py .\data\processed\D1 .\data\processed\D2 -ch .\data\processed\D1+D2\p1_output -e 100 -f 5 --cuda`.
      - `python .\src\models\model_1\train_test_crossed_fold.py .\data\processed\D1 .\data\processed\D3 -ch .\data\processed\D1+D3\p1_output -e 100 -f 5 --cuda`.
      - `python .\src\models\model_1\train_test_crossed_fold.py .\data\processed\D2 .\data\processed\D3 -ch .\data\processed\D2+D3\p1_output -e 100 -f 5 --cuda`.
      - `python .\src\models\model_1\train_test_crossed_fold.py .\data\processed\D1 .\data\processed\D2 .\data\processed\D3 -ch .\data\processed\D1+D2+D3\p1_output -e 100 -f 5 --cuda`.
  - Para Ubuntu:
      - `python ./src/models/model_1/train_test_crossed_fold.py ./data/processed/D1 -ch models/model1_predictions/D1 -e 100 -f 5 --cuda`.
      - `python ./src/models/model_1/train_test_crossed_fold.py ./data/processed/D2 -ch models/model1_predictions/D2 -e 100 -f 5 --cuda`.
      - `python ./src/models/model_1/train_test_crossed_fold.py ./data/processed/D3 -ch models/model1_predictions/D3 -e 100 -f 5 --cuda`.
      - `python ./src/models/model_1/train_test_crossed_fold.py ./data/processed/D1 ./data/processed/D2 -ch models/model1_predictions/D1+D2 -e 100 -f 5 --cuda`.
      - `python ./src/models/model_1/train_test_crossed_fold.py ./data/processed/D1 ./data/processed/D3 -ch models/model1_predictions/D1+D3 -e 100 -f 5 --cuda`.
      - `python ./src/models/model_1/train_test_crossed_fold.py ./data/processed/D2 ./data/processed/D3 -ch models/model1_predictions/D2+D3 -e 100 -f 5 --cuda`.
      - `python ./src/models/model_1/train_test_crossed_fold.py ./data/processed/D1 ./data/processed/D2 ./data/processed/D3 -ch models/model1_predictions/D1+D2+D3 -e 100 -f 5 --cuda`.

## Prototipo 2 - Original
1. Con `main_env` activado, ejecutar en Ubuntu:
    - `python ./src/models/model_2/main_fold.py ./data/processed/D1 -o ./models/model2_predictions/D1 -f 5 --cuda`
    - `python ./src/models/model_2/main_fold.py ./data/processed/D2 -o ./models/model2_predictions/D2 -f 5 --cuda`
    - `python ./src/models/model_2/main_fold.py ./data/processed/D3 -o ./models/model2_predictions/D3 -f 5 --cuda`
    - `python ./src/models/model_2/main_fold.py ./data/processed/D1 ./data/processed/D2 -o ./models/model2_predictions/D1D2 -f 5 --cuda`
    - `python ./src/models/model_2/main_fold.py ./data/processed/D1 ./data/processed/D3 -o ./models/model2_predictions/D1D3 -f 5 --cuda`
    - `python ./src/models/model_2/main_fold.py ./data/processed/D2 ./data/processed/D3 -o ./models/model2_predictions/D2D3 -f 5 --cuda`
    - `python ./src/models/model_2/main_fold.py ./data/processed/D1 ./data/processed/D2 ./data/processed/D3 -o ./models/model2_predictions/D1D2D3 -f 5 --cuda`

## Prototipo 2 - VGG16
1. Con `main_env` activado, ejecutar en Ubuntu:
    - `python ./src/models/model_2/main_fold.py ./data/processed/D1 -o ./models/model2_vgg16_predictions/D1 -f 5 --cuda -m "model2_vgg16" -lt 0.08 -sc 3.5` 
    - `python ./src/models/model_2/main_fold.py ./data/processed/D2 -o ./models/model2_vgg16_predictions/D2 -f 5 --cuda -m "model2_vgg16" -lt 0.08 -sc 3.5`
    - `python ./src/models/model_2/main_fold.py ./data/processed/D3 -o ./models/model2_vgg16_predictions/D3 -f 5 --cuda -m "model2_vgg16" -lt 0.08 -sc 3.5`
    - `python ./src/models/model_2/main_fold.py ./data/processed/D1 ./data/processed/D2 -o ./models/model2_vgg16_predictions/D1+D2 -f 5 --cuda -m "model2_vgg16" -lt 0.08 -sc 3.5`
    - `python ./src/models/model_2/main_fold.py ./data/processed/D1 ./data/processed/D3 -o ./models/model2_vgg16_predictions/D1+D3 -f 5 --cuda -m "model2_vgg16" -lt 0.08 -sc 3.5`
    - `python ./src/models/model_2/main_fold.py ./data/processed/D2 ./data/processed/D3 -o ./models/model2_vgg16_predictions/D2+D3 -f 5 --cuda -m "model2_vgg16" -lt 0.08 -sc 3.5`
    - `python ./src/models/model_2/main_fold.py ./data/processed/D1 ./data/processed/D2 ./data/processed/D3 -o ./models/model2_vgg16_predictions/D1+D2+D3 -f 5 --cuda -m "model2_vgg16" -lt 0.08 -sc 3.5`

## Prototipo 2 - Resnet50
1. Con `main_env` activado, ejecutar en Ubuntu:
    - `python ./src/models/model_2/main_fold.py ./data/processed/D1 -o ./models/model2_resnet50_predictions/D1 -f 5 --cuda -m "model2_resnet" -lt 0.08 -sc 3.5` 
    - `python ./src/models/model_2/main_fold.py ./data/processed/D2 -o ./models/model2_resnet50_predictions/D2 -f 5 --cuda -m "model2_resnet" -lt 0.08 -sc 3.5`
    - `python ./src/models/model_2/main_fold.py ./data/processed/D3 -o ./models/model2_resnet50_predictions/D3 -f 5 --cuda -m "model2_resnet" -lt 0.08 -sc 3.5`
    - `python ./src/models/model_2/main_fold.py ./data/processed/D1 ./data/processed/D2 -o ./models/model2_resnet50_predictions/D1+D2 -f 5 --cuda -m "model2_resnet" -lt 0.08 -sc 3.5`
    - `python ./src/models/model_2/main_fold.py ./data/processed/D1 ./data/processed/D3 -o ./models/model2_resnet50_predictions/D1+D3 -f 5 --cuda -m "model2_resnet" -lt 0.08 -sc 3.5`
    - `python ./src/models/model_2/main_fold.py ./data/processed/D2 ./data/processed/D3 -o ./models/model2_resnet50_predictions/D2+D3 -f 5 --cuda -m "model2_resnet" -lt 0.08 -sc 3.5`
    - `python ./src/models/model_2/main_fold.py ./data/processed/D1 ./data/processed/D2 ./data/processed/D3 -o ./models/model2_resnet50_predictions/D1+D2+D3 -f 5 --cuda -m "model2_resnet" -lt 0.08 -sc 3.5`

## Prototipo 2 - Nasnet Large
1. Con `main_env` activado, ejecutar en Ubuntu:
    - `python ./src/models/model_2/main_fold.py ./data/processed/D1 -o ./models/model2_nasnet_predictions/D1 -f 5 --cuda -m "model2_nasnet" -lt 0.08 -sc 3.5` 
    - `python ./src/models/model_2/main_fold.py ./data/processed/D2 -o ./models/model2_nasnet_predictions/D2 -f 5 --cuda -m "model2_nasnet" -lt 0.08 -sc 3.5`
    - `python ./src/models/model_2/main_fold.py ./data/processed/D3 -o ./models/model2_nasnet_predictions/D3 -f 5 --cuda -m "model2_nasnet" -lt 0.08 -sc 3.5`
    - `python ./src/models/model_2/main_fold.py ./data/processed/D1 ./data/processed/D2 -o ./models/model2_nasnet_predictions/D1+D2 -f 5 --cuda -m "model2_nasnet" -lt 0.08 -sc 3.5`
    - `python ./src/models/model_2/main_fold.py ./data/processed/D1 ./data/processed/D3 -o ./models/model2_nasnet_predictions/D1+D3 -f 5 --cuda -m "model2_nasnet" -lt 0.08 -sc 3.5`
    - `python ./src/models/model_2/main_fold.py ./data/processed/D2 ./data/processed/D3 -o ./models/model2_nasnet_predictions/D2+D3 -f 5 --cuda -m "model2_nasnet" -lt 0.08 -sc 3.5`
    - `python ./src/models/model_2/main_fold.py ./data/processed/D1 ./data/processed/D2 ./data/processed/D3 -o ./models/model2_nasnet_predictions/D1+D2+D3 -f 5 --cuda -m "model2_nasnet" -lt 0.08 -sc 3.5`

## Prototipo 2 - Inception V3
1. Con `main_env` activado, ejecutar en Ubuntu:
    - `python ./src/models/model_2/main_fold.py ./data/processed/D1 -o ./models/model2_inceptionv3_predictions/D1 -f 5 --cuda -m "model2_inceptionv3" -lt 0.08 -sc 3.5` 
    - `python ./src/models/model_2/main_fold.py ./data/processed/D2 -o ./models/model2_inceptionv3_predictions/D2 -f 5 --cuda -m "model2_inceptionv3" -lt 0.08 -sc 3.5`
    - `python ./src/models/model_2/main_fold.py ./data/processed/D3 -o ./models/model2_inceptionv3_predictions/D3 -f 5 --cuda -m "model2_inceptionv3" -lt 0.08 -sc 3.5`
    - `python ./src/models/model_2/main_fold.py ./data/processed/D1 ./data/processed/D2 -o ./models/model2_inceptionv3_predictions/D1+D2 -f 5 --cuda -m "model2_inceptionv3" -lt 0.08 -sc 3.5`
    - `python ./src/models/model_2/main_fold.py ./data/processed/D1 ./data/processed/D3 -o ./models/model2_inceptionv3_predictions/D1+D3 -f 5 --cuda -m "model2_inceptionv3" -lt 0.08 -sc 3.5`
    - `python ./src/models/model_2/main_fold.py ./data/processed/D2 ./data/processed/D3 -o ./models/model2_inceptionv3_predictions/D2+D3 -f 5 --cuda -m "model2_inceptionv3" -lt 0.08 -sc 3.5`
    - `python ./src/models/model_2/main_fold.py ./data/processed/D1 ./data/processed/D2 ./data/processed/D3 -o ./models/model2_inceptionv3_predictions/D1+D2+D3 -f 5 --cuda -m "model2_inceptionv3" -lt 0.08 -sc 3.5`

## Prototipo 2 - DenseNet
1. Con `main_env` activado, ejecutar en Ubuntu:
    - `python ./src/models/model_2/main_fold.py ./data/processed/D1 -o ./models/model2_densenet_predictions/D1 -f 5 --cuda -m "model2_densenet" -lt 0.08 -sc 3.5` 
    - `python ./src/models/model_2/main_fold.py ./data/processed/D2 -o ./models/model2_densenet_predictions/D2 -f 5 --cuda -m "model2_densenet" -lt 0.08 -sc 3.5`
    - `python ./src/models/model_2/main_fold.py ./data/processed/D3 -o ./models/model2_densenet_predictions/D3 -f 5 --cuda -m "model2_densenet" -lt 0.08 -sc 3.5`
    - `python ./src/models/model_2/main_fold.py ./data/processed/D1 ./data/processed/D2 -o ./models/model2_densenet_predictions/D1+D2 -f 5 --cuda -m "model2_densenet" -lt 0.08 -sc 3.5`
    - `python ./src/models/model_2/main_fold.py ./data/processed/D1 ./data/processed/D3 -o ./models/model2_densenet_predictions/D1+D3 -f 5 --cuda -m "model2_densenet" -lt 0.08 -sc 3.5`
    - `python ./src/models/model_2/main_fold.py ./data/processed/D2 ./data/processed/D3 -o ./models/model2_densenet_predictions/D2+D3 -f 5 --cuda -m "model2_densenet" -lt 0.08 -sc 3.5`
    - `python ./src/models/model_2/main_fold.py ./data/processed/D1 ./data/processed/D2 ./data/processed/D3 -o ./models/model2_densenet_predictions/D1+D2+D3 -f 5 --cuda -m "model2_densenet" -lt 0.08 -sc 3.5`


## Prototipo 3
1. Con el environment "main_env" activado ejecutar:
    - `python ./src/models/model_3/train_predict_model.py ./data/processed/D1 -ch ./models/model3_predictions/D1 -f 5 --cuda -b_s 32 -e 100 -f 5` 
    - `python ./src/models/model_3/train_predict_model.py ./data/processed/D2 -ch ./models/model3_predictions/D2 -f 5 --cuda -b_s 32 -e 100 -f 5`
    - `python ./src/models/model_3/train_predict_model.py ./data/processed/D3 -ch ./models/model3_predictions/D3 -f 5 --cuda -b_s 32 -e 100 -f 5`
    - `python ./src/models/model_3/train_predict_model.py ./data/processed/D1 ./data/processed/D2 -ch ./models/model3_predictions/D1+D2 -f 5 --cuda -b_s 32 -e 100 -f 5`
    - `python ./src/models/model_3/train_predict_model.py ./data/processed/D1 ./data/processed/D3 -ch ./models/model3_predictions/D1+D3 -f 5 --cuda -b_s 32 -e 100 -f 5`
    - `python ./src/models/model_3/train_predict_model.py ./data/processed/D2 ./data/processed/D3 -ch ./models/model3_predictions/D2+D3 -f 5 --cuda -b_s 32 -e 100 -f 5`
    - `python ./src/models/model_3/train_predict_model.py ./data/processed/D1 ./data/processed/D2 ./data/processed/D3 -ch ./models/model3_predictions/D1+D2+D3 -f 5 --cuda -b_s 32 -e 100 -f 5`


## Obtener metricas
Desde la raiz del proyecto ejecutar:
- `python ./src/models/analysis/metrics.py -f 5 -m "model_1"`
- `python ./src/models/analysis/metrics.py -f 5 -m "model2_densenet"`
- `python ./src/models/analysis/metrics.py -f 5 -m "model2_inceptionv3"`
- `python ./src/models/analysis/metrics.py -f 5 -m "model2_nasnet"`
- `python ./src/models/analysis/metrics.py -f 5 -m "model2"`
- `python ./src/models/analysis/metrics.py -f 5 -m "model2_resnet50"`
- `python ./src/models/analysis/metrics.py -f 5 -m "model2_vgg16"`
- `python ./src/models/analysis/metrics.py -f 5 -m "model3"`
