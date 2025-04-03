import numpy as np
import pandas as pd
import cv2
import os
import random

# Cargar los datos
y_pred = np.load(r"D:\D\outputs\P2\D1\vfd_y_pred_fold_8.npy")
y_labels = np.load(r"D:\D\outputs\P2\D1\vfd_y_test_fold_8.npy")

# Cargar el dataframe
complete_df = pd.read_csv(
    r"D:\D\D1\forgery_info\forgery_info.csv",
    usecols=['name', 'labels', 'deletion_point_i'],
    dtype={'name': str, 'labels': int, 'deletion_point_i': str}
)
total_videos = len(complete_df)
Y = complete_df['labels'].astype(int).to_numpy()

# Configurar el StratifiedKFold
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
folds = list(skf.split(np.arange(total_videos), Y))

# Obtener los índices del fold 8
train_idx, test_idx = folds[7]  # Fold 8 (índice es 7 porque Python empieza en 0)

# Mapear los índices de los casos
indices = {'verdadero_positivo': None, 'falso_positivo': None,
           'falso_negativo': None, 'verdadero_negativo': None}

for i, (pred, label) in enumerate(zip(y_pred, y_labels)):
    if i in test_idx:  # Asegurarse de usar solo índices del fold 8
        if label == 1 and pred == 1:
            indices['verdadero_positivo'] = i
        elif label == 0 and pred == 1:
            indices['falso_positivo'] = i
        elif label == 1 and pred == 0:
            indices['falso_negativo'] = i
        elif label == 0 and pred == 0:
            indices['verdadero_negativo'] = i

# Función para guardar frames
def save_frames(video_path, frame_indices, output_prefix, output_dir):
    cap = cv2.VideoCapture(video_path)
    for i, frame_index in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            output_path = os.path.join(output_dir, f"{output_prefix}_{i+1}.png")
            cv2.imwrite(output_path, frame)
    cap.release()

# Crear la carpeta de frames si no existe
output_dir = r"D:\D\D1\frames"
os.makedirs(output_dir, exist_ok=True)

# Procesar cada caso
for case, index in indices.items():
    if index is not None:
        video_name = complete_df.iloc[index]['name']
        video_path = os.path.join(r"D:\D\D1\forgery_data", video_name)
        deletion_point = complete_df.iloc[index]['deletion_point_i']
        if deletion_point == '-':  # Caso original (no manipulado)
            total_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = sorted(random.sample(range(total_frames), 3))
        else:  # Caso manipulado
            manipulated_frame = int(deletion_point.split(',')[0][1:])
            frame_indices = [manipulated_frame - 1, manipulated_frame, manipulated_frame + 1]

        save_frames(video_path, frame_indices, case, output_dir)
