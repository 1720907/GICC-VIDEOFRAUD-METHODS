import argparse
import moviepy.editor as mpe
import os

def convert_to_seconds(time_str):
    minutes = int(time_str[:-2])
    seconds = int(time_str[-2:])
    total_seconds = minutes * 60 + seconds
    return total_seconds

def procesar_tuplas(tuplas, data_path, data_export):
    filename = os.path.basename(data_path)
    filename = os.path.splitext(filename)[0]
    video = mpe.VideoFileClip(data_path)
    i=0
    for tupla in tuplas:
        i+=1
        valor1, valor2 = tupla
        valor1 = convert_to_seconds(valor1)
        valor2 = convert_to_seconds(valor2)
        sub_vid = video.subclip(valor1,valor2)
        sub_vid.write_videofile(f"{data_export}/{filename}_{i}.mp4")


def main():
    parser = argparse.ArgumentParser(description='Procesar tuplas de dos valores.')
    parser.add_argument('tuplas', metavar='T', type=str, nargs='+', help='Tuplas de dos valores')
    parser.add_argument('--archivo', '-a', type=str, help='Ubicación del archivo')
    parser.add_argument('--save_path', '-sp', type=str, help='Ubicación donde guardar videos de salida')
    args = parser.parse_args()
    
    # Verificamos si el número de elementos en tuplas es par
    if len(args.tuplas) % 2 != 0:
        print("El número de elementos debe ser par")
        return
    
    # Convertimos la lista de argumentos en una lista de tuplas de dos elementos
    tuplas = [(args.tuplas[i], args.tuplas[i+1]) for i in range(0, len(args.tuplas), 2)]
    
    # Procesamos las tuplas
    procesar_tuplas(tuplas, args.archivo, args.save_path)
    print("Tuplas procesadas")


if __name__ == "__main__":
    main()