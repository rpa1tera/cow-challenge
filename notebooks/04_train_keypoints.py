
import os

os.environ["WANDB_DISABLED"] = "true"
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns" 
from ultralytics import YOLO, settings


try:
    settings.update({"mlflow": False, "wandb": False, "clearml": False, "comet": False, "tensorboard": False})
except Exception as e:
    print(f"Aviso: Não foi possível atualizar as configurações: {e}")

import sys

def train_model():
    # Carregar um modelo
    try:
        model = YOLO('yolo26n-pose.pt')  # carregar um modelo pré-treinado
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        print("Certifique-se de que 'ultralytics' está instalado e você tem acesso à internet para baixar os pesos.")
        return

    # Treinar o modelo
    
    print("Iniciando treinamento...")
    results = model.train(
        data='g:/PYTHON/cow/data/cow-pose.yaml',
        epochs=90,
        imgsz=640,
        batch=8, 
        project='g:/PYTHON/cow/models',
        name='yolov26n-cow-pose',
        exist_ok=True, 
        amp=False, 
        # --- DATA AUGMENTATION
        hsv_h=0.015, # Variação natural de tom (cor/hue)
        hsv_s=0.9,   # Variação EXTREMA de saturação (90%)
        hsv_v=0.8    # Variação EXTREMA de luminosidade/brilho (80%)
    )
    
    print("Treinamento concluído.")
    print(f"Melhor modelo salvo em {results.save_dir}")

if __name__ == "__main__":
    train_model()