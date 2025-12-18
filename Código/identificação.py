import numpy as np
from sklearn.model_selection import KFold
import os
import datetime
from sklearn.preprocessing import LabelEncoder

timestamp = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

# Carrega os dados
descritor = "HOG"  # "HOG" OU "LBP"
data = np.load(f"Código/descritores_{descritor}.npz")

vetores = data["vetores"]
rotulos = data["rotulos"]
ids_unicos = data["ids_unicos"]
tipo = data["tipo"]

print("Arquivo carregado:", tipo)
print(f"Número de classes (IDs): {len(ids_unicos)}")
print(f"Número de amostras: {len(vetores)}")

# Codificar labels para 0, 1, 2, ...
le = LabelEncoder()
rotulos_encoded = le.fit_transform(rotulos)
num_classes = len(le.classes_)

print(f"Classes codificadas: {num_classes}")

# Configurações
modelo = "mlp"  # "linear" ou "mlp"
pasta_base = f"Resultados_Identificacao/{descritor}/{modelo}"
os.makedirs(pasta_base, exist_ok=True)

arquivo_config = os.path.join(pasta_base, "run_config.txt")
arquivo_error = os.path.join(pasta_base, "run_error.txt")
arquivo_model = os.path.join(pasta_base, "model.dat")

with open(arquivo_config, "w") as f:
    f.write(f"Execucao em {timestamp}\n")

with open(arquivo_error, "w") as f:
    f.write(f"Execucao em {timestamp}\n")

# Funções de inicialização aprimoradas
def inicializar_weights_he(input_dim, output_dim):
    """He initialization para ReLU"""
    std = np.sqrt(2.0 / input_dim)
    return np.random.randn(output_dim, input_dim) * std

def inicializar_weights_xavier(input_dim, output_dim):
    """Xavier/Glorot initialization"""
    std = np.sqrt(2.0 / (input_dim + output_dim))
    return np.random.randn(output_dim, input_dim) * std

# K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
acuracias = []
precisoes = []
recalls = []
f1_scores = []

melhor_fold = (-1, -np.inf)
pior_fold = (-1, np.inf)

for fold_id, (treino_idx, teste_idx) in enumerate(kf.split(vetores)):
    print(f"\n{'='*60}")
    print(f"FOLD {fold_id + 1}/5")
    print('='*60)
    
    # Separação dos dados
    X_treino = vetores[treino_idx]
    y_treino = rotulos_encoded[treino_idx]
    X_teste = vetores[teste_idx]
    y_teste = rotulos_encoded[teste_idx]
    
    # Balanceamento de classes - análise
    unique_train, counts_train = np.unique(y_treino, return_counts=True)
    unique_test, counts_test = np.unique(y_teste, return_counts=True)
    
    print(f"Amostras treino: {len(X_treino)} (mín: {counts_train.min()}, máx: {counts_train.max()})")
    print(f"Amostras teste: {len(X_teste)} (mín: {counts_test.min()}, máx: {counts_test.max()})")
    
    # Normalização Z-score por fold
    media = X_treino.mean(axis=0)
    desvio = X_treino.std(axis=0) + 1e-8
    
    X_treino = (X_treino - media) / desvio
    X_teste = (X_teste - media) / desvio
    
    n_atrib = X_treino.shape[1]
    
    # Histórico para logging
    historico_epocas = []
    
    if modelo == "linear":
        # Modelo Linear com regularização
        W = inicializar_weights_xavier(n_atrib, num_classes)
        b = np.zeros(num_classes)
        
        # Hiperparâmetros
        taxa = 0.01
        decaimento = 1e-4  # Regularização L2
        épocas = 100
        tam_batch = 64
        melhor_val_loss = np.inf
        paciência = 15
        piora = 0
        
        for ep in range(épocas):
            # Shuffle
            perm = np.random.permutation(len(X_treino))
            X_treino_shuffled = X_treino[perm]
            y_treino_shuffled = y_treino[perm]
            
            losses_batch = []
            
            for i in range(0, len(X_treino_shuffled), tam_batch):
                Xb = X_treino_shuffled[i:i+tam_batch]
                yb = y_treino_shuffled[i:i+tam_batch]
                
                # Forward pass
                logits = Xb @ W.T + b
                exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exp_logits / (exp_logits.sum(axis=1, keepdims=True) + 1e-8)
                
                # Loss com regularização L2
                loss = -np.mean(np.log(probs[np.arange(len(yb)), yb] + 1e-8))
                reg_loss = decaimento * 0.5 * np.sum(W * W)
                total_loss = loss + reg_loss
                losses_batch.append(total_loss)
                
                # Backward pass
                grad = probs.copy()
                grad[np.arange(len(yb)), yb] -= 1
                grad /= len(yb)
                
                dW = grad.T @ Xb + decaimento * W
                db = grad.sum(axis=0)
                
                W -= taxa * dW
                b -= taxa * db
            
            treino_loss = np.mean(losses_batch)
            
            # Validação
            logits_val = X_teste @ W.T + b
            exp_val = np.exp(logits_val - np.max(logits_val, axis=1, keepdims=True))
            probs_val = exp_val / (exp_val.sum(axis=1, keepdims=True) + 1e-8)
            
            val_loss = -np.mean(np.log(probs_val[np.arange(len(y_teste)), y_teste] + 1e-8))
            
            # Early stopping
            if val_loss < melhor_val_loss:
                melhor_val_loss = val_loss
                piora = 0
                # Salvar melhor modelo
                W_best, b_best = W.copy(), b.copy()
            else:
                piora += 1
            
            # Log do histórico
            historico_epocas.append((ep, treino_loss, val_loss))
            
            if ep % 10 == 0:
                pred_val = probs_val.argmax(axis=1)
                acur_val = (pred_val == y_teste).mean()
                print(f"Época {ep:3d} | Loss Treino: {treino_loss:.4f} | Loss Val: {val_loss:.4f} | Acurácia Val: {acur_val:.4f}")
            
            if piora >= paciência:
                print(f"Early stopping na época {ep}")
                W, b = W_best, b_best
                break
        
        # Avaliação final
        logits_final = X_teste @ W.T + b
        exp_final = np.exp(logits_final - np.max(logits_final, axis=1, keepdims=True))
        probs_final = exp_final / (exp_final.sum(axis=1, keepdims=True) + 1e-8)
        
    else:  # MLP
        # Arquitetura: Input -> Hidden1 -> Dropout -> Hidden2 -> Output
        hidden1 = 64
        hidden2 = 16
        
        # Inicialização com He para ReLU
        W1 = inicializar_weights_he(n_atrib, hidden1)
        b1 = np.zeros(hidden1)
        W2 = inicializar_weights_he(hidden1, hidden2)
        b2 = np.zeros(hidden2)
        W3 = inicializar_weights_xavier(hidden2, num_classes)
        b3 = np.zeros(num_classes)
        
        # Hiperparâmetros
        taxa = 0.001
        decaimento = 1e-5  # Regularização L2
        épocas = 150
        tam_batch = 64
        dropout_rate = 0.5
        
        melhor_val_loss = np.inf
        melhor_val_acc = 0
        paciência = 20
        piora = 0
        
        for ep in range(épocas):
            # Shuffle
            perm = np.random.permutation(len(X_treino))
            X_treino_shuffled = X_treino[perm]
            y_treino_shuffled = y_treino[perm]
            
            losses_batch = []
            
            for i in range(0, len(X_treino_shuffled), tam_batch):
                Xb = X_treino_shuffled[i:i+tam_batch]
                yb = y_treino_shuffled[i:i+tam_batch]
                
                # Forward pass com Dropout
                # Camada 1
                z1 = Xb @ W1.T + b1
                a1 = np.maximum(0, z1)  # ReLU
                # Dropout durante treino
                if dropout_rate > 0:
                    mask1 = np.random.binomial(1, 1-dropout_rate, size=a1.shape) / (1-dropout_rate)
                    a1 *= mask1
                
                # Camada 2
                z2 = a1 @ W2.T + b2
                a2 = np.maximum(0, z2)
                if dropout_rate > 0:
                    mask2 = np.random.binomial(1, 1-dropout_rate, size=a2.shape) / (1-dropout_rate)
                    a2 *= mask2
                
                # Camada de saída
                logits = a2 @ W3.T + b3
                exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exp_logits / (exp_logits.sum(axis=1, keepdims=True) + 1e-8)
                
                # Loss com regularização L2
                loss = -np.mean(np.log(probs[np.arange(len(yb)), yb] + 1e-8))
                reg_loss = decaimento * 0.5 * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))
                total_loss = loss + reg_loss
                losses_batch.append(total_loss)
                
                # Backward pass
                grad3 = probs.copy()
                grad3[np.arange(len(yb)), yb] -= 1
                grad3 /= len(yb)
                
                dW3 = grad3.T @ a2 + decaimento * W3
                db3 = grad3.sum(axis=0)
                
                grad2 = (grad3 @ W3) * (z2 > 0)
                if dropout_rate > 0:
                    grad2 *= mask2
                dW2 = grad2.T @ a1 + decaimento * W2
                db2 = grad2.sum(axis=0)
                
                grad1 = (grad2 @ W2) * (z1 > 0)
                if dropout_rate > 0:
                    grad1 *= mask1
                dW1 = grad1.T @ Xb + decaimento * W1
                db1 = grad1.sum(axis=0)
                
                # Atualização
                W1 -= taxa * dW1
                b1 -= taxa * db1
                W2 -= taxa * dW2
                b2 -= taxa * db2
                W3 -= taxa * dW3
                b3 -= taxa * db3
            
            treino_loss = np.mean(losses_batch)
            
            # Validação (sem dropout)
            z1_val = X_teste @ W1.T + b1
            a1_val = np.maximum(0, z1_val)
            z2_val = a1_val @ W2.T + b2
            a2_val = np.maximum(0, z2_val)
            logits_val = a2_val @ W3.T + b3
            
            exp_val = np.exp(logits_val - np.max(logits_val, axis=1, keepdims=True))
            probs_val = exp_val / (exp_val.sum(axis=1, keepdims=True) + 1e-8)
            
            val_loss = -np.mean(np.log(probs_val[np.arange(len(y_teste)), y_teste] + 1e-8))
            pred_val = probs_val.argmax(axis=1)
            val_acc = (pred_val == y_teste).mean()
            
            # Early stopping baseado em acurácia
            if val_acc > melhor_val_acc:
                melhor_val_acc = val_acc
                melhor_val_loss = val_loss
                piora = 0
                # Salvar melhor modelo
                W1_best, b1_best = W1.copy(), b1.copy()
                W2_best, b2_best = W2.copy(), b2.copy()
                W3_best, b3_best = W3.copy(), b3.copy()
            else:
                piora += 1
            
            # Log do histórico
            historico_epocas.append((ep, treino_loss, val_loss))
            
            if ep % 10 == 0:
                print(f"Época {ep:3d} | Loss Treino: {treino_loss:.4f} | Loss Val: {val_loss:.4f} | Acurácia Val: {val_acc:.4f}")
            
            if piora >= paciência:
                print(f"Early stopping na época {ep}")
                W1, b1 = W1_best, b1_best
                W2, b2 = W2_best, b2_best
                W3, b3 = W3_best, b3_best
                break
        
        # Avaliação final
        z1_final = X_teste @ W1.T + b1
        a1_final = np.maximum(0, z1_final)
        z2_final = a1_final @ W2.T + b2
        a2_final = np.maximum(0, z2_final)
        logits_final = a2_final @ W3.T + b3
        
        exp_final = np.exp(logits_final - np.max(logits_final, axis=1, keepdims=True))
        probs_final = exp_final / (exp_final.sum(axis=1, keepdims=True) + 1e-8)
    
    # Avaliação do fold
    pred = probs_final.argmax(axis=1)
    acur = (pred == y_teste).mean()
    acuracias.append(acur)
    
    # Cálculo de métricas adicionais
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    precisao = precision_score(y_teste, pred, average='weighted', zero_division=0)
    recall = recall_score(y_teste, pred, average='weighted', zero_division=0)
    f1 = f1_score(y_teste, pred, average='weighted', zero_division=0)
    
    precisoes.append(precisao)
    recalls.append(recall)
    f1_scores.append(f1)
    
    print(f"\nResultados Fold {fold_id}:")
    print(f"Acurácia: {acur:.4f}")
    print(f"Precisão: {precisao:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Atualizar melhor e pior fold
    if acur > melhor_fold[1]:
        melhor_fold = (fold_id, acur)
    if acur < pior_fold[1]:
        pior_fold = (fold_id, acur)
    
    # Salvar histórico no arquivo de erro
    with open(arquivo_error, "a") as f:
        f.write(f"\n=== FOLD {fold_id} ===\n")
        f.write("Época;Loss_Treino;Loss_Val\n")
        for ep, loss_treino, loss_val in historico_epocas:
            f.write(f"{ep};{loss_treino:.6f};{loss_val:.6f}\n")
        f.write(f"\nResultados Fold {fold_id}:\n")
        f.write(f"Acurácia: {acur:.4f}\n")
        f.write(f"Precisão: {precisao:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")

# Resultados finais
print(f"\n{'='*60}")
print("RESULTADOS FINAIS DA VALIDAÇÃO CRUZADA")
print('='*60)

print(f"\nAcurácias por fold: {[f'{acc:.4f}' for acc in acuracias]}")
print(f"Acurácia média: {np.mean(acuracias):.4f} ± {np.std(acuracias):.4f}")

print(f"\nPrecisões por fold: {[f'{p:.4f}' for p in precisoes]}")
print(f"Precisão média: {np.mean(precisoes):.4f} ± {np.std(precisoes):.4f}")

print(f"\nRecalls por fold: {[f'{r:.4f}' for r in recalls]}")
print(f"Recall médio: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")

print(f"\nF1-Scores por fold: {[f'{f:.4f}' for f in f1_scores]}")
print(f"F1-Score médio: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

print(f"\nMelhor fold: {melhor_fold[0]} com acurácia {melhor_fold[1]:.4f}")
print(f"Pior fold: {pior_fold[0]} com acurácia {pior_fold[1]:.4f}")

# Salvar configuração final
with open(arquivo_config, "a") as f:
    f.write(f"\nDescritor: {descritor}\n")
    f.write(f"Modelo: {modelo}\n")
    f.write(f"Número de classes: {num_classes}\n")
    f.write(f"Número de atributos: {n_atrib}\n")
    
    f.write(f"\nHiperparâmetros:\n")
    if modelo == "linear":
        f.write(f"Taxa de aprendizado: {taxa}\n")
        f.write(f"Regularização L2: {decaimento}\n")
        f.write(f"Épocas máximas: {épocas}\n")
        f.write(f"Tamanho do batch: {tam_batch}\n")
        f.write(f"Paciência early stopping: {paciência}\n")
    else:
        f.write(f"Taxa de aprendizado: {taxa}\n")
        f.write(f"Regularização L2: {decaimento}\n")
        f.write(f"Dropout rate: {dropout_rate}\n")
        f.write(f"Tamanho Hidden1: {hidden1}\n")
        f.write(f"Tamanho Hidden2: {hidden2}\n")
        f.write(f"Épocas máximas: {épocas}\n")
        f.write(f"Tamanho do batch: {tam_batch}\n")
        f.write(f"Paciência early stopping: {paciência}\n")
    
    f.write(f"\nResultados:\n")
    f.write(f"Acurácias: {[f'{acc:.4f}' for acc in acuracias]}\n")
    f.write(f"Acurácia média: {np.mean(acuracias):.4f}\n")
    f.write(f"Desvio padrão acurácia: {np.std(acuracias):.4f}\n")
    
    f.write(f"\nPrecisões: {[f'{p:.4f}' for p in precisoes]}\n")
    f.write(f"Precisão média: {np.mean(precisoes):.4f}\n")
    
    f.write(f"\nRecalls: {[f'{r:.4f}' for r in recalls]}\n")
    f.write(f"Recall médio: {np.mean(recalls):.4f}\n")
    
    f.write(f"\nF1-Scores: {[f'{f:.4f}' for f in f1_scores]}\n")
    f.write(f"F1-Score médio: {np.mean(f1_scores):.4f}\n")
    
    f.write(f"\nMelhor fold: {melhor_fold}\n")
    f.write(f"Pior fold: {pior_fold}\n")