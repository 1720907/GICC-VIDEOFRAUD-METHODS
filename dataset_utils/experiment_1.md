# Introduction
This experiment applies 10-fold cross-validation with P1 and P2 on datasets D1, D2, and D3 (and their combinations). The outputs will be `ypred` and `ylabels` for each case, which will then be processed by a comparative algorithm.

## Dataset
1. Git clone or git pull: `git clone https://github.com/pshiguihara/GICC_VideoFraud_Prototype`.
2. Download the following datasets:
   - D1: `wget "https://usilpe-my.sharepoint.com/:u:/g/personal/jorge_ceron_usil_pe/EcywWa3ML15Fk1NIGOSOqOoB4ApVcD-OC_a2kgsVeXlyJg?e=mNL2XW&download=1" -O D1.zip`
   - D2: `wget "https://usilpe-my.sharepoint.com/:u:/g/personal/jorge_ceron_usil_pe/ESxm4Ezr7X5JiiF8Dcq9CwYBO8zeZ3QB-IxdMTx5VaAxdQ?e=ZWuG43&download=1" -O D2.zip`
   - D3: `wget "https://usilpe-my.sharepoint.com/:u:/g/personal/jorge_ceron_usil_pe/Eb0AEAqLX2NLh5KlNJVztWYBEUh9yLwXjf598JUmV0jPRg?e=aKP5Kl&download=1" -O D3.zip`
3. Extract them into the project's root directory.
4. Run `bash main.sh` to create folders for the remaining datasets (D1+D2, D2+D3, D1+D3, D1+D2+D3).

## Prototype 1
1. Activate the `env_1` environment.
2. Run train-val-test:
   - `python prototype_1/train_test_crossed_fold.py D1/ -ch D1/p1_output/ -e 10 -f 10 --cuda`
   - `python prototype_1/train_test_crossed_fold.py D2/ -ch D2/p1_output/ -e 10 -f 10 --cuda`
   - `python prototype_1/train_test_crossed_fold.py D3/ -ch D3/p1_output/ -e 10 -f 10 --cuda`
   - `python prototype_1/train_test_crossed_fold.py D1/ D2/ -ch D1+D2/p1_output/ -e 10 -f 10 --cuda`
   - `python prototype_1/train_test_crossed_fold.py D1/ D3/ -ch D1+D3/p1_output/ -e 10 -f 10 --cuda`
   - `python prototype_1/train_test_crossed_fold.py D2/ D3/ -ch D2+D3/p1_output/ -e 10 -f 10 --cuda`
   - `python prototype_1/train_test_crossed_fold.py D1/ D2/ D3/ -ch D1+D2+D3/p1_output/ -e 10 -f 10 --cuda`

## Prototype 2
1. Activate the `env_2` environment.
2. Run:
   - `python prototype_2/main_fold.py D1/ -g D1/p2_graphs -o D1/p2_output --cuda`
   - `python prototype_2/main_fold.py D2/ -g D2/p2_graphs -o D2/p2_output --cuda`
   - `python prototype_2/main_fold.py D3/ -g D3/p2_graphs -o D3/p2_output --cuda`
   - `python prototype_2/main_fold.py D1/ D2/ -g D1+D2/p2_graphs -o D1+D2/p2_output --cuda`
   - `python prototype_2/main_fold.py D2/ D3/ -g D2+D3/p2_graphs -o D2+D3/p2_output --cuda`
   - `python prototype_2/main_fold.py D1/ D3/ -g D1+D3/p2_graphs -o D1+D3/p2_output --cuda`
   - `python prototype_2/main_fold.py D1/ D2/ D3/ -g D1+D2+D3/p2_graphs -o D1+D2+D3/p2_output --cuda`