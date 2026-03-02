import os

# Desativar integrações via variáveis de ambiente simples
os.environ["WANDB_DISABLED"] = "true"
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"

from ultralytics import YOLO, settings

# Desativar explicitamente via API de configurações
try:
    settings.update({"mlflow": False, "wandb": False, "clearml": False, "comet": False, "tensorboard": False})
except Exception as e:
    print(f"Aviso: Não foi possível atualizar as configurações: {e}")

def evaluate_test_set():
    print("Iniciando avaliação no conjunto de teste (Dataset Inédito)...")
    
    # Carregar o melhor modelo treinado
    model_path = 'g:/PYTHON/cow/models/yolov26n-cow-pose/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"Erro: Modelo não encontrado em {model_path}")
        return

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Erro ao carregar o modelo YOLO: {e}")
        return

    # Executar a validação forçando o split='test' e forçando o salvamento de arquivos
    results = model.val(
        data='g:/PYTHON/cow/data/cow-pose.yaml',
        split='test',
        project='g:/PYTHON/cow/models',
        name='yolov26n-cow-pose-test_eval',
        exist_ok=True,
        save=True,       # Salva as imagens preditas com as caixas
        save_json=True,  # Salva os resultados das previsões em JSON
        save_txt=True,   # Salva as labels/previsões em .txt (formato yolo)
        plots=True       # Força a geração dos gráficos (PR curve, Conf matrix, etc)
    )
    
    print("\nAvaliação de Teste Concluída!")
    print(f"Resultados detalhados (imagens e CSV) salvos em: {results.save_dir}")

if __name__ == "__main__":
    evaluate_test_set()
