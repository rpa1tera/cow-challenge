import os
import cv2
import numpy as np
import shutil
from pathlib import Path

def apply_gaussian_noise(image, var_limit=(10.0, 50.0)):
    """
    Aplica ruído gaussiano a uma imagem.
    var_limit: Define a variação máx e min (intensidade do ruido)
    """
    row, col, ch = image.shape
    mean = 0
    
    # Sorteia uma variância aleatória dentro do limite para cada imagem
    var = np.random.uniform(var_limit[0], var_limit[1])
    sigma = var ** 0.5
    
    # Gera o ruído
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    
    # Adiciona à imagem e clampa os valores entre 0 e 255
    noisy = image.astype(np.float32) + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    return noisy

def offline_augmentation():
    base_dir = Path('g:/PYTHON/cow/data/processed')
    
    # Vamos aumentar apenas os dados de TREINAMENTO
    # Nunca se aumenta Validação ou Teste (para manter os testes reais)
    train_images_dir = base_dir / 'images' / 'train'
    train_labels_dir = base_dir / 'labels' / 'train'
    
    print(f"Iniciando Augmentation Offline em: {train_images_dir}")
    
    # Pegar imagens originais (ignorando as que já tem _noise no nome para não rodar infinito)
    images = [f for f in train_images_dir.glob("*.jpg") if "_noise" not in f.name]
    
    augmented_count = 0
    
    for img_path in images:
        label_path = train_labels_dir / (img_path.stem + ".txt")
        
        # Se por acaso não tiver label, ignora
        if not label_path.exists():
            continue
            
        # Pular se já estiver aumentada
        new_img_name = img_path.stem + "_noise.jpg"
        new_label_name = img_path.stem + "_noise.txt"
        
        new_img_path = train_images_dir / new_img_name
        new_label_path = train_labels_dir / new_label_name
        
        if new_img_path.exists() and new_label_path.exists():
            continue
            
        # 1. Carrega e aplica o Ruído
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        noisy_img = apply_gaussian_noise(img, var_limit=(15.0, 45.0))
        
        # 2. Salva a nova imagem
        cv2.imwrite(str(new_img_path), noisy_img)
        
        # 3. Copia o Label exato (Afinal as coordenadas dos ossos não mudaram!)
        shutil.copy2(str(label_path), str(new_label_path))
        
        augmented_count += 1
        
    print(f"Sucesso! {augmented_count} imagens de treinamento e seus gabaritos foram duplicados com Ruído Gaussiano.")

if __name__ == "__main__":
    offline_augmentation()
