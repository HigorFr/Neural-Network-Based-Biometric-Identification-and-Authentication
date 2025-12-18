import numpy as np
from sklearn.model_selection import KFold
import os
import datetime

timestamp = datetime.datetime.now().strftime("%d/%m/%Y %H:%M") 

# Carrega os npz que o extrator HOG/LBP.py gerou
descritor = "HOG"  # Colocar HOG OU LBP
data = np.load(f"Código/descritores_{descritor}.npz")

vetores = data["vetores"]
rotulos = data["rotulos"]
ids_unicos = data["ids_unicos"]
tipo = data["tipo"]

print("Arquivo carregado:", tipo)
print("Número real de IDs carregados dos descritores:", len(ids_unicos))


# ===============================
#  Autenticação – gerar pares
# ===============================

def gerar_pares(vetores, rotulos):
    rng = np.random.default_rng(42)
    X_pairs = []
    y_pairs = []

    # Mapa rótulo → lista de índices
    mapa = {}
    for idx, r in enumerate(rotulos):
        mapa.setdefault(r, []).append(idx)

    todas_ids = list(mapa.keys())
    total_ids = len(todas_ids)

    print("IDs realmente presentes nos descritores:", total_ids)

    for r in todas_ids:
        idxs = mapa[r]
        if len(idxs) < 2:
            continue

        # positivo
        a, b = rng.choice(idxs, size=2, replace=False)
        X_pairs.append(np.concatenate([vetores[a], vetores[b]]))
        y_pairs.append(1)

        # negativo
        outras = [x for x in todas_ids if x != r]
        r2 = rng.choice(outras)
        b2 = rng.choice(mapa[r2])
        X_pairs.append(np.concatenate([vetores[a], vetores[b2]]))
        y_pairs.append(0)

    return np.array(X_pairs), np.array(y_pairs)

X, y = gerar_pares(vetores, rotulos)
print("Total de pares gerados:", len(X))


# ===============================
#  Pastas e arquivos de saída
# ===============================
modelo = "linear"   # "linear" ou "mlp"

pasta_base = f"Resultados_Autenticacao/{descritor}/{modelo}"
os.makedirs(pasta_base, exist_ok=True)

arquivo_config = os.path.join(pasta_base, "run_config.txt")
arquivo_error  = os.path.join(pasta_base, "run_error.txt")
arquivo_model  = os.path.join(pasta_base, "model.dat")

with open(arquivo_config, "w") as f:
    f.write(f"Execucao em {timestamp}\n")

with open(arquivo_error, "w") as f:
    f.write(f"Execucao em {timestamp}\n")


# ===============================
#  K-FOLD
# ===============================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
acuracias = []

melhor_fold = (-1, -np.inf)
pior_fold   = (-1,  np.inf)


for fold_id, (treino_idx, teste_idx) in enumerate(kf.split(X)):

    X_treino = X[treino_idx]
    y_treino = y[treino_idx]
    X_teste  = X[teste_idx]
    y_teste  = y[teste_idx]

    # Normaliza
    media = X_treino.mean(axis=0)
    desvio = X_treino.std(axis=0) + 1e-6

    X_treino = (X_treino - media) / desvio
    X_teste  = (X_teste  - media) / desvio

    n_atrib = X_treino.shape[1]
    num_classes = 2  # binário

    # histórico para salvar no txt
    historico_epocas = []


    # =============================
    #  CLASSIFICADOR LINEAR
    # =============================
    if modelo == "linear":

        W = np.random.randn(num_classes, n_atrib) * 0.01
        b = np.zeros(num_classes)

        taxa = 0.01
        épocas = 15
        tam_batch = 64

        for ep in range(épocas):

            perm = np.random.permutation(len(X_treino))
            X_treino = X_treino[perm]
            y_treino = y_treino[perm]

            losses_batch = []

            for i in range(0, len(X_treino), tam_batch):
                Xb = X_treino[i:i+tam_batch]
                yb = y_treino[i:i+tam_batch]

                logits = Xb @ W.T + b
                e = np.exp(logits - logits.max(axis=1, keepdims=True))
                probs = e / e.sum(axis=1, keepdims=True)

                loss = -np.log(np.clip(probs[np.arange(len(yb)), yb], 1e-12, 1)).mean()
                losses_batch.append(loss)

                grad = probs
                grad[np.arange(len(yb)), yb] -= 1
                grad /= len(yb)

                W -= taxa * (grad.T @ Xb)
                b -= taxa * grad.sum(axis=0)

            treino_loss = np.mean(losses_batch)

            # VALIDAÇÃO
            logits_t = X_teste @ W.T + b
            e_val = np.exp(logits_t - logits_t.max(axis=1, keepdims=True))
            probs_val = e_val / e_val.sum(axis=1, keepdims=True)

            val_loss = -np.log(np.clip(probs_val[np.arange(len(y_teste)), y_teste], 1e-12, 1)).mean()

            historico_epocas.append((ep, treino_loss, val_loss))

            with open(arquivo_error, "a") as f:
                f.write(f"{ep};{treino_loss};{val_loss}\n")


    # =============================
    #  MLP
    # =============================
    else:

        H = 128
        W1 = np.random.randn(H, n_atrib) * 0.01
        b1 = np.zeros(H)
        W2 = np.random.randn(num_classes, H) * 0.01
        b2 = np.zeros(num_classes)

        taxa = 0.001
        épocas = 200
        pac = 64
        paciência = 3
        piora = 0
        melhor_valor = np.inf

        for ep in range(épocas):

            perm = np.random.permutation(len(X_treino))
            X_treino = X_treino[perm]
            y_treino = y_treino[perm]

            losses_batch = []

            for i in range(0, len(X_treino), pac):
                Xb = X_treino[i:i+pac]
                yb = y_treino[i:i+pac]

                h = Xb @ W1.T + b1
                h_relu = np.maximum(0, h)
                logits = h_relu @ W2.T + b2

                e = np.exp(logits - logits.max(axis=1, keepdims=True))
                probs = e / e.sum(axis=1, keepdims=True)

                loss = -np.log(np.clip(probs[np.arange(len(yb)), yb], 1e-12, 1)).mean()
                losses_batch.append(loss)

                grad2 = probs
                grad2[np.arange(len(yb)), yb] -= 1
                grad2 /= len(yb)

                grad_W2 = grad2.T @ h_relu
                grad_b2 = grad2.sum(axis=0)

                grad_h = grad2 @ W2
                grad_h[h <= 0] = 0

                grad_W1 = grad_h.T @ Xb
                grad_b1 = grad_h.sum(axis=0)

                W1 -= taxa * grad_W1
                b1 -= taxa * grad_b1
                W2 -= taxa * grad_W2
                b2 -= taxa * grad_b2

            treino_loss = np.mean(losses_batch)

            # VALIDAÇÃO
            h_t = X_teste @ W1.T + b1
            h_t = np.maximum(0, h_t)
            logits_t = h_t @ W2.T + b2

            e_val = np.exp(logits_t - logits_t.max(axis=1, keepdims=True))
            probs_val = e_val / e_val.sum(axis=1, keepdims=True)

            val_loss = -np.log(np.clip(probs_val[np.arange(len(y_teste)), y_teste], 1e-12, 1)).mean()

            historico_epocas.append((ep, treino_loss, val_loss))

            with open(arquivo_error, "a") as f:
                f.write(f"{ep};{treino_loss};{val_loss}\n")

            # early stopping
            if val_loss < melhor_valor:
                melhor_valor = val_loss
                piora = 0
            else:
                piora += 1
                if piora >= paciência:
                    break


    pred = probs_val.argmax(axis=1)
    acur = (pred == y_teste).mean()
    acuracias.append(acur)

    if acur > melhor_fold[1]:
        melhor_fold = (fold_id, acur)
    if acur < pior_fold[1]:
        pior_fold = (fold_id, acur)


with open(arquivo_config, "a") as f:
    f.write(f"\nDescritor: {descritor}\n")
    f.write(f"Modelo: {modelo}\n")
    f.write(f"Acuracias: {acuracias}\n")
    f.write(f"Acuracia media: {np.mean(acuracias):.4f}\n")
    f.write(f"Melhor fold: {melhor_fold}\n")
    f.write(f"Pior fold: {pior_fold}\n")

print("Acurácias por fold:", acuracias)
print("Acurácia média:", np.mean(acuracias))
