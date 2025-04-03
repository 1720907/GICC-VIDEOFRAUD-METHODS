# Introduction

- Este experimento consiste en aplicar 10-fold-cross-validation con P1 y P2 en los conjuntos de datos D1, D2 y D3 (y sus combinaciones). Las salidas serán ypred e ylabels para cada caso, para finalmente pasarlas por un algoritmo comparativo.

## Dataset

1. Git clone o git pull: `git clone https://github.com/pshiguihara/GICC_VideoFraud_Prototype`.
2. Descargar siguientes datasets:
  - D1: `wget "https://usilpe-my.sharepoint.com/:u:/g/personal/jorge_ceron_usil_pe/EcywWa3ML15Fk1NIGOSOqOoB4ApVcD-OC_a2kgsVeXlyJg?e=mNL2XW&download=1" -O D1.zip`
  - D2: `wget "https://usilpe-my.sharepoint.com/:u:/g/personal/jorge_ceron_usil_pe/ESxm4Ezr7X5JiiF8Dcq9CwYBO8zeZ3QB-IxdMTx5VaAxdQ?e=ZWuG43&download=1" -O D2.zip`
  - D3: `wget "https://usilpe-my.sharepoint.com/:u:/g/personal/jorge_ceron_usil_pe/Eb0AEAqLX2NLh5KlNJVztWYBEUh9yLwXjf598JUmV0jPRg?e=aKP5Kl&download=1" -O D3.zip`
3. Descromprimirlos en la raiz del proyecto.
4. Correr `bash main.sh` para que se creen las carpetas de los datasets restantes (D1+D2, D2+D3, D1+D3, D1+D2+D3).

## Prototipo 1

1. Activar environment `env_1`.
2. Ejecutar train-val-test:
  - `python prototype_1/train_test_crossed_fold.py D1/ -ch D1/p1_outuput/ -e 10 -f 10 --cuda`.
  - `python prototype_1/train_test_crossed_fold.py D2/ -ch D2/p1_outuput/ -e 10 -f 10 --cuda`.
  - `python prototype_1/train_test_crossed_fold.py D3/ -ch D3/p1_outuput/ -e 10 -f 10 --cuda`.
  - `python prototype_1/train_test_crossed_fold.py D1/ D2/ -ch D1+D2/p1_outuput/ -e 10 -f 10 --cuda`.
  - `python prototype_1/train_test_crossed_fold.py D1/ D3/ -ch D1+D3/p1_outuput/ -e 10 -f 10 --cuda`.
  - `python prototype_1/train_test_crossed_fold.py D2/ D3/ -ch D2+D3/p1_outuput/ -e 10 -f 10 --cuda`.
  - `python prototype_1/train_test_crossed_fold.py D1/ D2/ D3/ -ch D1+D2+D3/p1_outuput/ -e 10 -f 10 --cuda`.


## Prototipo 2
1. Activar environment `env_2`.
2. Ejecutar: 
- `python prototype_2/main_fold.py D1/ -g D1/p2_graphs -o D1/p2_output --cuda`.
- `python prototype_2/main_fold.py D2/ -g D2/p2_graphs -o D2/p2_output --cuda`.
- `python prototype_2/main_fold.py D3/ -g D3/p2_graphs -o D3/p2_output --cuda`.
- `python prototype_2/main_fold.py D1/ D2/ -g D1+D2/p2_graphs -o D1+D2/p2_output --cuda`.
- `python prototype_2/main_fold.py D2/ D3/ -g D2+D3/p2_graphs -o D2+D3/p2_output --cuda`.
- `python prototype_2/main_fold.py D1/ D3/ -g D1+D3/p2_graphs -o D1+D3/p2_output --cuda`.
- `python prototype_2/main_fold.py D1/ D2/ D3/ -g D1+D2+D3/p2_graphs -o D1+D2+D3/p2_output --cuda`.
