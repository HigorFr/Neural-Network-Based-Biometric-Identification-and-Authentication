import numpy as np
from sklearn.model_selection import KFold
import os
import datetime

timestamp = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

# Carrega os npz
descritor = "HOG"  # "HOG" OU "LBP"
data = np.load(f"Código/descritores_{descritor}.npz")

vetores = data["vetores"]
rotulos = data["rotulos"]
ids_unicos = data["ids_unicos"]
tipo = data["tipo"]

print("Arquivo carregado:", tipo)
print("Número real de IDs:", len(ids_unicos))

#Função para gerar pares
def gerar_pares_balanceados(vetores, rotulos, pares_positivos=12, pares_negativos=12):
    rng = np.random.default_rng(42)
    X_pairs = []
    y_pairs = []
    
    mapa = {}
    for idx, r in enumerate(rotulos):
        mapa.setdefault(r, []).append(idx)
    
    ids = list(mapa.keys())
    
    for r in ids:
        idxs = mapa[r]
        if len(idxs) < 2:
            continue
        
        # Pares positivos
        for _ in range(pares_positivos):
            a, b = rng.choice(idxs, size=2, replace=False)
            # Diferença absoluta e também produto (similaridade)
            diff = np.abs(vetores[a] - vetores[b])
            prod = vetores[a] * vetores[b]  # Similaridade
            pair_features = np.concatenate([diff, prod])
            X_pairs.append(pair_features)
            y_pairs.append(1)
        
        # Pares negativos (garantindo balanceamento)
        for _ in range(pares_negativos):
            a = rng.choice(idxs)
            r2 = rng.choice([x for x in ids if x != r])
            b = rng.choice(mapa[r2])
            diff = np.abs(vetores[a] - vetores[b])
            prod = vetores[a] * vetores[b]
            pair_features = np.concatenate([diff, prod])
            X_pairs.append(pair_features)
            y_pairs.append(0)
    
    return np.array(X_pairs), np.array(y_pairs)

# Inicialização aprimorada (He initialization para ReLU)
def inicializar_weights_he(input_dim, output_dim):
    std = np.sqrt(2.0 / input_dim)
    return np.random.randn(output_dim, input_dim) * std

def inicializar_weights_xavier(input_dim, output_dim):
    std = np.sqrt(2.0 / (input_dim + output_dim))
    return np.random.randn(output_dim, input_dim) * std

# Gera pares (AUMENTEI features concatenando diff e prod)
X_pairs, y_pairs = gerar_pares_balanceados(vetores, rotulos)
print("Total de pares gerados:", len(X_pairs))
print("Shape dos pares:", X_pairs.shape)

# Configurações
modelo = "mlp"  # "linear" ou "mlp"
pasta_base = f"Resultados_Autenticacao/{descritor}/{modelo}"
os.makedirs(pasta_base, exist_ok=True)

arquivo_config = os.path.join(pasta_base, "run_config.txt")
arquivo_error  = os.path.join(pasta_base, "run_error.txt")

with open(arquivo_config, "w") as f:
    f.write(f"Execucao em {timestamp}\n")
    f.write(f"Descritor: {descritor}\nModelo: {modelo}\n")

# K-FOLD (com normalização DENTRO de cada fold)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
acuracias = []
melhor_fold = (-1, -np.inf)
pior_fold = (-1, np.inf)

for fold_id, (treino_idx, teste_idx) in enumerate(kf.split(X_pairs)):
    print(f"\n=== Fold {fold_id} ===")
    
    #normaliza
    X_treino = X_pairs[treino_idx]
    y_treino = y_pairs[treino_idx]
    X_teste  = X_pairs[teste_idx]
    y_teste  = y_pairs[teste_idx]
    
    #Normalização Z-score por fold (evita data leakage)
    media_treino = X_treino.mean(axis=0)
    std_treino = X_treino.std(axis=0) + 1e-8
    X_treino = (X_treino - media_treino) / std_treino
    X_teste = (X_teste - media_treino) / std_treino
    
    n_atrib = X_treino.shape[1]
    num_classes = 2
    
    if modelo == "linear":
        #Modelo Linear com L2 Regularization
        W = inicializar_weights_xavier(n_atrib, num_classes)
        b = np.zeros(num_classes)
        
        taxa = 0.01
        decaimento = 1e-4  # L2 regularization
        épocas = 100
        tam_batch = 64
        
        melhor_val_loss = np.inf
        paciência = 15
        piora = 0
        
        for ep in range(épocas):
            perm = np.random.permutation(len(X_treino))
            X_treino = X_treino[perm]
            y_treino = y_treino[perm]
            
            losses = []
            for i in range(0, len(X_treino), tam_batch):
                Xb = X_treino[i:i+tam_batch]
                yb = y_treino[i:i+tam_batch]
                
                # Forward
                logits = Xb @ W.T + b
                exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exp_logits / (exp_logits.sum(axis=1, keepdims=True) + 1e-8)
                
                # Loss com regularização L2
                loss = -np.mean(np.log(probs[np.arange(len(yb)), yb] + 1e-8))
                reg_loss = decaimento * 0.5 * np.sum(W * W)
                total_loss = loss + reg_loss
                losses.append(total_loss)
                
                # Backprop
                grad = probs.copy()
                grad[np.arange(len(yb)), yb] -= 1
                grad /= len(yb)
                
                dW = grad.T @ Xb + decaimento * W
                db = grad.sum(axis=0)
                
                W -= taxa * dW
                b -= taxa * db
            
            # Validação
            logits_val = X_teste @ W.T + b
            exp_val = np.exp(logits_val - np.max(logits_val, axis=1, keepdims=True))
            probs_val = exp_val / (exp_val.sum(axis=1, keepdims=True) + 1e-8)
            val_loss = -np.mean(np.log(probs_val[np.arange(len(y_teste)), y_teste] + 1e-8))
            
            # Early stopping
            if val_loss < melhor_val_loss:
                melhor_val_loss = val_loss
                piora = 0
                # Salvar melhor modelo do fold
                W_best, b_best = W.copy(), b.copy()
            else:
                piora += 1
            
            if piora >= paciência:
                print(f"Early stopping na época {ep}")
                W, b = W_best, b_best
                break
    


    else:  # MLP
        # Arquitetura: Input -> Hidden (128) -> Hidden (64) -> Output
        hidden1 = 64
        hidden2 = 16
        
        # Inicialização He para ReLU
        W1 = inicializar_weights_he(n_atrib, hidden1)
        b1 = np.zeros(hidden1)
        W2 = inicializar_weights_he(hidden1, hidden2)
        b2 = np.zeros(hidden2)
        W3 = inicializar_weights_xavier(hidden2, num_classes)
        b3 = np.zeros(num_classes)
        
        taxa = 0.001
        decaimento = 1e-5
        épocas = 150
        tam_batch = 64
        
        melhor_val_loss = np.inf
        paciência = 20
        piora = 0
        
        for ep in range(épocas):
            perm = np.random.permutation(len(X_treino))
            X_treino_shuffled = X_treino[perm]
            y_treino_shuffled = y_treino[perm]
            
            losses = []
            for i in range(0, len(X_treino_shuffled), tam_batch):
                Xb = X_treino_shuffled[i:i+tam_batch]
                yb = y_treino_shuffled[i:i+tam_batch]
                
                # Forward pass
                # Camada 1
                z1 = Xb @ W1.T + b1
                a1 = np.maximum(0, z1)  # ReLU
                # Dropout (50%)
                dropout_mask1 = (np.random.rand(*a1.shape) > 0.5) / 0.5
                a1 *= dropout_mask1
                
                # Camada 2
                z2 = a1 @ W2.T + b2
                a2 = np.maximum(0, z2)
                dropout_mask2 = (np.random.rand(*a2.shape) > 0.5) / 0.5
                a2 *= dropout_mask2
                
                # Output
                logits = a2 @ W3.T + b3
                exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exp_logits / (exp_logits.sum(axis=1, keepdims=True) + 1e-8)
                
                # Loss com L2
                loss = -np.mean(np.log(probs[np.arange(len(yb)), yb] + 1e-8))
                reg_loss = decaimento * 0.5 * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))
                total_loss = loss + reg_loss
                losses.append(total_loss)
                
                # Backpropagation
                grad3 = probs.copy()
                grad3[np.arange(len(yb)), yb] -= 1
                grad3 /= len(yb)
                
                dW3 = grad3.T @ a2 + decaimento * W3
                db3 = grad3.sum(axis=0)
                
                grad2 = (grad3 @ W3) * (z2 > 0) * dropout_mask2
                dW2 = grad2.T @ a1 + decaimento * W2
                db2 = grad2.sum(axis=0)
                
                grad1 = (grad2 @ W2) * (z1 > 0) * dropout_mask1
                dW1 = grad1.T @ Xb + decaimento * W1
                db1 = grad1.sum(axis=0)
                
                # Atualização com momentum (simples)
                W1 -= taxa * dW1
                b1 -= taxa * db1
                W2 -= taxa * dW2
                b2 -= taxa * db2
                W3 -= taxa * dW3
                b3 -= taxa * db3
            
            # Validação (sem dropout)
            z1_val = X_teste @ W1.T + b1
            a1_val = np.maximum(0, z1_val)
            z2_val = a1_val @ W2.T + b2
            a2_val = np.maximum(0, z2_val)
            logits_val = a2_val @ W3.T + b3
            
            exp_val = np.exp(logits_val - np.max(logits_val, axis=1, keepdims=True))
            probs_val = exp_val / (exp_val.sum(axis=1, keepdims=True) + 1e-8)
            val_loss = -np.mean(np.log(probs_val[np.arange(len(y_teste)), y_teste] + 1e-8))
            
            # Early stopping
            if val_loss < melhor_val_loss:
                melhor_val_loss = val_loss
                piora = 0
                # Salvar melhor modelo
                W1_best, b1_best = W1.copy(), b1.copy()
                W2_best, b2_best = W2.copy(), b2.copy()
                W3_best, b3_best = W3.copy(), b3.copy()
            else:
                piora += 1
            
            if piora >= paciência:
                print(f"Early stopping na época {ep}")
                W1, b1 = W1_best, b1_best
                W2, b2 = W2_best, b2_best
                W3, b3 = W3_best, b3_best
                break
    
    # Avaliação final
    if modelo == "linear":
        logits_final = X_teste @ W.T + b
    else:
        z1_final = X_teste @ W1.T + b1
        a1_final = np.maximum(0, z1_final)
        z2_final = a1_final @ W2.T + b2
        a2_final = np.maximum(0, z2_final)
        logits_final = a2_final @ W3.T + b3
    
    probs_final = np.exp(logits_final - np.max(logits_final, axis=1, keepdims=True))
    probs_final = probs_final / (probs_final.sum(axis=1, keepdims=True) + 1e-8)
    pred = probs_final.argmax(axis=1)
    
    acur = (pred == y_teste).mean()
    acuracias.append(acur)
    
    # Métricas adicionais
    tp = ((pred == 1) & (y_teste == 1)).sum()
    tn = ((pred == 0) & (y_teste == 0)).sum()
    fp = ((pred == 1) & (y_teste == 0)).sum()
    fn = ((pred == 0) & (y_teste == 1)).sum()
    
    precisao = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precisao * recall) / (precisao + recall + 1e-8)
    
    print(f"Acurácia: {acur:.4f}")
    print(f"Precisão: {precisao:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    if acur > melhor_fold[1]:
        melhor_fold = (fold_id, acur)
    if acur < pior_fold[1]:
        pior_fold = (fold_id, acur)

# Resultados finais
print(f"\n{'='*50}")
print("RESULTADOS FINAIS:")
print(f"Acurácias por fold: {acuracias}")
print(f"Acurácia média: {np.mean(acuracias):.4f} ± {np.std(acuracias):.4f}")
print(f"Melhor fold: {melhor_fold}")
print(f"Pior fold: {pior_fold}")

# Salva configuração final
with open(arquivo_config, "a") as f:
    f.write(f"\nAcurácias: {acuracias}\n")
    f.write(f"Acurácia média: {np.mean(acuracias):.4f}\n")
    f.write(f"Desvio padrão: {np.std(acuracias):.4f}\n")
    f.write(f"Melhor fold: {melhor_fold}\n")
    f.write(f"Pior fold: {pior_fold}\n")