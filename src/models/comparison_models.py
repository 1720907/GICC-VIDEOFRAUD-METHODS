import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_rel

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-m1','--model_1', type=str, default='./checkpoint', help='Path containing model 1 predictions and real labels')
  parser.add_argument('-m2','--model_2', type=str, default='./checkpoint', help='Path containing model 2 predictions and real labels')
  return parser.parse_args()

def run():
  args = parse_args()
  path_m1 = args.model_1
  path_m2 = args.model_2

  path_predictions1 = os.path.join(path_m1, "y_pred.npy")
  path_y_test1 = os.path.join(path_m1, "y_test.npy")
  predictions1 = np.load(path_predictions1)
  y_test_1 = np.load(path_y_test1)

  path_predictions2 = os.path.join(path_m2, "y_pred.npy")
  path_y_test2 = os.path.join(path_m2, "y_test.npy")
  predictions2 = np.load(path_predictions2)
  y_test_2 = np.load(path_y_test2)

  accuracyA = accuracy_score(y_test_1, predictions1)
  accuracyB = accuracy_score(y_test_2, predictions2)

  t_estadistico, p_value = ttest_rel(predicciones_A, predicciones_B)
  print("Accuracy del Modelo 1: ", accuracyA)
  print("Accuracy del Modelo 2: ", accuracyB)
  print("Valor p: ", p_value)

  threshold = 0.05 #5% signigicance
  if p_value <= threshold:
    print("Hipótesis nula rechazada")
  else:
    print("Hipótesis nula NO rechazada")



if __name__ == '__main__':
  inicio = time.time()
  run()
  fin = time.time()
  execution_time = fin - inicio
  hours = int(execution_time // 3600)
  minutes = int((execution_time % 3600) // 60)
  seconds = int(execution_time % 60)

  print(f"Time of execution: {hours}H:{minutes}M:{seconds}S")