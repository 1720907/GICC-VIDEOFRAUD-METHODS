import argparse
import time
import numpy as np
from os.path import join
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

dataset_dict = ['D1', 'D2', 'D3', 'D1+D2', 'D1+D3', 'D2+D3', 'D1+D2+D3']

model_paths = {
    "model1": "./models/model1_predictions",
    "model2_densenet": "./models/model2_densenet_predictions",
    "model2_inceptionv3": "./models/model2_inceptionv3_predictions",
    "model2_nasnet": "./models/model2_nasnet_predictions",
    "model2": "./models/model2_predictions",
    "model2_resnet50": "./models/model2_resnet50_predictions",
    "model2_vgg16": "./models/model2_vgg16_predictions",
    "model3": "./models/model3_predictions"
}

def get_model_path(model_name):
    return model_paths.get(model_name, "Modelo no encontrado")

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-m','--model_name', type=str, default='', help='Name of the dataset')
  parser.add_argument('-f','--folds', type=int, default=5, help='Number of folds')
  return parser.parse_args()

def show_report_p2(fold, dataset, model_path):
    vfd_pred = np.load(join(model_path, dataset, f"vfd_y_pred_fold_{fold}.npy"))
    vfd_test = np.load(join(model_path, dataset, f"vfd_y_test_fold_{fold}.npy"))
    print(classification_report(vfd_test, vfd_pred, labels = (0,1), output_dict=False, zero_division=0))
    return classification_report(vfd_test, vfd_pred, labels = (0,1), output_dict=True, zero_division=0), vfd_pred, vfd_test

def show_report_p1(fold, dataset, model_path):
    y_pred = np.load(join(model_path,dataset,f"y_pred_fold_{fold}.npy"))
    y_test = np.load(join(model_path,dataset,f"y_test_fold_{fold}.npy"))
    return classification_report(y_test, y_pred, output_dict=True, zero_division=0)

def find_metrics(n_folds, dataset, model_path, function):
    precision = []
    recall = []
    f1 = []
    acc = []

    for i in range(n_folds):
        report, vfd_pred, vfd_test = function((i+1), dataset, model_path)
        precision.append(report['macro avg']['precision'])
        recall.append(report['macro avg']['recall'])
        f1.append(report['macro avg']['f1-score'])
        # Compute accuracy separately
        fold_accuracy = accuracy_score(vfd_test, vfd_pred)
        acc.append(fold_accuracy)

    # Calculate means
    mean_precision = np.mean(precision)
    mean_recall = np.mean(recall)
    mean_f1 = np.mean(f1)
    mean_acc = np.mean(acc)

    # Calculate standard deviations
    std_precision = np.std(precision)
    std_recall = np.std(recall)
    std_f1 = np.std(f1)
    std_acc = np.std(acc)

    # Display results
    print(f"Results {dataset}: ")
    print(f"Mean Precision (Macro Avg): {mean_precision:.2f} \u00b1 {std_precision:.2f}")
    print(f"Mean Recall (Macro Avg): {mean_recall:.2f} \u00b1 {std_recall:.2f}")
    print(f"Mean F1-Score (Macro Avg): {mean_f1:.2f} \u00b1 {std_f1:.2f}")
    print(f"Mean Accuracy: {mean_acc:.2f} \u00b1 {std_acc:.2f}")


def run():
    args = parse_args()
    model_name = args.model_name
    num_folds = args.folds
    print(f'Results: {model_name}: ')
    model_path = get_model_path(model_name)
    

    for element in dataset_dict:
        find_metrics(num_folds, element, model_path, show_report_p2)

if __name__ == '__main__':
    inicio = time.time()
    run()
    fin = time.time()
    execution_time = fin - inicio
    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    seconds = int(execution_time % 60)

    print(f"Time of execution: {hours}H:{minutes}M:{seconds}S")