
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import os
import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

def get_model(model_name, strategy=3):
    if model_name == "mlp":
        # Strategy 1/2: 4 hidden layers de 170.
        # Strategy 3: 4 hidden layers de 200.
        if strategy in [1, 2]:
            hidden_layers = (170, 170, 170, 170)
            alpha_l2 = 0.00100
        else:
            hidden_layers = (200, 200, 200, 200)
            alpha_l2 = 0.00001
            
        return MLPClassifier(
            hidden_layer_sizes=hidden_layers, 
            activation='tanh',
            alpha=alpha_l2,  # L2 (Ridge) regularization 
            max_iter=2000, 
            random_state=42
        )
    elif model_name == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "svm":
        return SVC(kernel='rbf', probability=True, random_state=42)
    elif model_name == "xgboost":
        if not HAS_XGB:
            raise ImportError("XGBoost não está instalado. Execute `pip install xgboost`.")
        return XGBClassifier(n_estimators=100, eval_metric='mlogloss', random_state=42)
    elif model_name == "catboost":
        if not HAS_CATBOOST:
            raise ImportError("CatBoost não está instalado. Execute `pip install catboost`.")
        return CatBoostClassifier(iterations=100, random_state=42, verbose=0)
    elif model_name == "lightgbm":
        if not HAS_LGBM:
            raise ImportError("LightGBM não está instalado. Execute `pip install lightgbm`.")
        return LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")

def train_classifier(model_name="xgboost", strategy=3, mode="both"):
    if mode == "geo":
        feature_file = 'g:/PYTHON/cow/data/processed/cow_features_geo.csv'
    else:
        feature_file = 'g:/PYTHON/cow/data/processed/cow_features_hybrid.csv'
    
    if not os.path.exists(feature_file):
        print(f"Feature file not found: {feature_file}. Run feature extraction first.")
        return

    df = pd.read_csv(feature_file)
    print(f"Loaded features for {len(df)} images.")
    
    # Preprocessing
    initial_len = len(df)
    df = df.dropna()
    print(f"Dropped {initial_len - len(df)} rows with missing features.")
    
    if len(df) < 2:
        print("Not enough data to train.")
        return

    # Features and Target
    all_feature_cols = [c for c in df.columns if c not in ["filename", "cow_id"]]
    
    img_cols = ["ratio_white_fur", "ratio_black_fur", "texture_contrast"]
    if mode == "geo":
        present_img_cols = []
    else:
        present_img_cols = [c for c in img_cols if c in all_feature_cols]
    
    if strategy == 1:
        feature_cols = [c for c in all_feature_cols if c.startswith("S1_")] + present_img_cols
        print(f"Usando Estratégia 1: 13 Features da Rump Area (F1-F13) {'+ Padrão de Pelagem' if mode == 'both' else ''}.")
    elif strategy == 2:
        feature_cols = [c for c in all_feature_cols if c.startswith("S2_")] + present_img_cols
        print(f"Usando Estratégia 2: 16 Features da Dorsal Area {'+ Padrão de Pelagem' if mode == 'both' else ''}.")
    else:
        feature_cols = [c for c in all_feature_cols if c.startswith("S")] + present_img_cols
        print(f"Usando Estratégia 3: 29 Features combinadas (S1 + S2) {'+ Padrão de Pelagem' if mode == 'both' else ''}.")
        
    X = df[feature_cols]
    y = df["cow_id"]
    
    # Encode Target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Classes: {le.classes_}")
    
    # Scaling (Crucial para Redes Neurais)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Instantiate Model
    print(f"\nInitializing model: {model_name.upper()}")
    clf = get_model(model_name, strategy=strategy)
    
    # Avaliação - Cross Validation
    scores = cross_val_score(clf, X_scaled, y_encoded, cv=5)
    print(f"\nModel Acurácia via 5-Fold Cross Validation: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    except ValueError:
        print("Warning: Some classes have only 1 sample, cannot do stratified split. Using random split.")
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
        
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Acurácia no Teste: {acc:.2f}")
    print("\nRelatório de Classificação (Teste):")
    
    # Para o classification_report, nem sempre todas as classes do conjunto existem no y_test
    # Passar os labels ativamente presentes:
    present_labels = sorted(list(set(y_test)))
    present_names = le.inverse_transform(present_labels)
    
    # Cast target_names to strings as classification_report expects a list of strings
    string_names = [str(name) for name in present_names]
    print(classification_report(y_test, y_pred, labels=present_labels, target_names=string_names))
    
    # Feature Importance
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        print("\nTop 10 Feature Importances:")
        print(feature_importance_df.head(10))
    else:
        print(f"\n[Nota]: O modelo {model_name} não fornece 'feature_importances_' diretamente da mesma forma que os baseados em árvore.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinar um modelo de classificação para identificação de vacas.")
    parser.add_argument(
        "--model", 
        type=str, 
        default="xgboost", 
        choices=["mlp", "random_forest", "svm", "xgboost", "catboost", "lightgbm"],
        help="Modelo para treinar: mlp, random_forest, svm, xgboost, catboost, lightgbm"
    )
    parser.add_argument(
        "--strategy", 
        type=int, 
        default=3, 
        choices=[1, 2, 3],
        help="Estratégia do Estudo 1 (1: S1/Garupa, 2: S2/Dorso, 3: Ambas)"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        default="both", 
        choices=["geo", "both"],
        help="Modo de extração usado (determina o CSV de entrada e features)"
    )
    args = parser.parse_args()
    
    train_classifier(model_name=args.model, strategy=args.strategy, mode=args.mode)
