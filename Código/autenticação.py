import numpy as np
from sklearn.model_selection import KFold
import os
import datetime


# =========================================================
# GERAÇÃO DE PARES BALANCEADOS
# =========================================================
def gerar_pares(vetores, rotulos, pares_positivos=12, pares_negativos=12):
    rng = np.random.default_rng(random_state)
    X_pairs, y_pairs = [], []

    mapa = {}
    for i, r in enumerate(rotulos):
        mapa.setdefault(r, []).append(i)

    ids = list(mapa.keys())

    for r in ids:
        idxs = mapa[r]
        if len(idxs) < 2:
            continue

        for _ in range(pares_positivos):
            a, b = rng.choice(idxs, size=2, replace=False)
            diff = np.abs(vetores[a] - vetores[b])
            prod = vetores[a] * vetores[b]
            X_pairs.append(np.concatenate([diff, prod]))
            y_pairs.append(1)

        for _ in range(pares_negativos):
            a = rng.choice(idxs)
            r2 = rng.choice([x for x in ids if x != r])
            b = rng.choice(mapa[r2])
            diff = np.abs(vetores[a] - vetores[b])
            prod = vetores[a] * vetores[b]
            X_pairs.append(np.concatenate([diff, prod]))
            y_pairs.append(0)

    return np.array(X_pairs), np.array(y_pairs)

def inicializar_weights_he(inp, out):
    return np.random.randn(out, inp) * np.sqrt(2.0 / inp)

def inicializar_weights_xavier(inp, out):
    return np.random.randn(out, inp) * np.sqrt(2.0 / (inp + out))








# =========================================================
# CONFIGURAÇÕES GERAIS
# =========================================================
timestamp = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
descritores = ["HOG","LBP"]  # "HOG" ou "LBP"
modelos = ["linear", "mlp"]
random_state = 42



for descritor in descritores:

    data = np.load(f"Código/descritores_{descritor}.npz")

    vetores = data["vetores"]
    rotulos = data["rotulos"]
    ids_unicos = data["ids_unicos"]
    tipo = data["tipo"]

    print("Arquivo carregado:", tipo)
    print("Número real de IDs:", len(ids_unicos))


    # =========================================================
    # PREPARAÇÃO DOS DADOS
    # =========================================================
    X_pairs, y_pairs = gerar_pares(vetores, rotulos)
    print("Total de pares:", len(X_pairs))
    print("Dimensão dos pares:", X_pairs.shape)

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    # =========================================================
    # LOOP DOS MODELOS
    # =========================================================
    for modelo in modelos:

        print(f"\n{'='*60}")
        print(f"MODELO: {modelo.upper()}")
        print(f"{'='*60}")

        pasta_base = f"Resultados_Autenticacao/{descritor}/{modelo}"
        os.makedirs(pasta_base, exist_ok=True)

        arquivo_config = os.path.join(pasta_base, "run_config.txt")
        arquivo_error = os.path.join(pasta_base, "run_error.txt")

        with open(arquivo_config, "w", encoding="utf-8") as f:
            f.write(f"Execução: {timestamp}\nDescritor: {descritor}\nModelo: {modelo}\n")

        with open(arquivo_error, "w", encoding="utf-8") as f:
            f.write(f"Execucao em {timestamp}\n")


        acuracias = []
        melhor_fold = (-1, -np.inf)
        pior_fold = (-1, np.inf)

        # =====================================================
        # K-FOLD
        # =====================================================
        for fold_id, (treino_idx, teste_idx) in enumerate(kf.split(X_pairs)):
            print(f"\n--- Fold {fold_id} ---")

            X_treino = X_pairs[treino_idx]
            y_treino = y_pairs[treino_idx]
            X_teste = X_pairs[teste_idx]
            y_teste = y_pairs[teste_idx]

            # Normalização Z-score por fold
            mu = X_treino.mean(axis=0)
            sigma = X_treino.std(axis=0) + 1e-8
            X_treino = (X_treino - mu) / sigma
            X_teste = (X_teste - mu) / sigma

            n_atrib = X_treino.shape[1]
            num_classes = 2

            #Linear
            if modelo == "linear":
                W = inicializar_weights_xavier(n_atrib, num_classes)
                b = np.zeros(num_classes)

                lr, l2, epocas, batch = 0.01, 1e-4, 100, 64
                melhor_loss, paciencia, piora = np.inf, 15, 0

                for ep in range(epocas):
                    perm = np.random.permutation(len(X_treino))
                    X_treino, y_treino = X_treino[perm], y_treino[perm]

                    losses = []   # <-- TEM QUE SER AQUI

                    for i in range(0, len(X_treino), batch):
                        Xb = X_treino[i:i+batch]
                        yb = y_treino[i:i+batch]

                        logits = Xb @ W.T + b
                        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
                        probs = exp / (exp.sum(axis=1, keepdims=True) + 1e-8)

                        loss = -np.mean(np.log(probs[np.arange(len(yb)), yb] + 1e-8))
                        total_loss = loss + 0.5 * l2 * np.sum(W * W)

                        losses.append(total_loss)

                        grad = probs
                        grad[np.arange(len(yb)), yb] -= 1
                        grad /= len(yb)

                        W -= lr * (grad.T @ Xb + l2 * W)
                        b -= lr * grad.sum(axis=0)

                    erro_treino = np.mean(losses)

                    logits_val = X_teste @ W.T + b
                    exp_val = np.exp(logits_val - logits_val.max(axis=1, keepdims=True))
                    probs_val = exp_val / (exp_val.sum(axis=1, keepdims=True) + 1e-8)
                    val_loss = -np.mean(np.log(probs_val[np.arange(len(y_teste)), y_teste] + 1e-8))

                    with open(arquivo_error, "a", encoding="utf-8") as f:
                        f.write(f"{ep};{erro_treino:.8f};{val_loss:.8f}\n")

                    if val_loss < melhor_loss:
                        melhor_loss, piora = val_loss, 0
                        W_best, b_best = W.copy(), b.copy()
                    else:
                        piora += 1

                    if piora >= paciencia:
                        W, b = W_best, b_best
                        break

                    logits_final = X_teste @ W.T + b




            else:
                h1, h2 = 64, 16
                W1, b1 = inicializar_weights_he(n_atrib, h1), np.zeros(h1)
                W2, b2 = inicializar_weights_he(h1, h2), np.zeros(h2)
                W3, b3 = inicializar_weights_xavier(h2, num_classes), np.zeros(num_classes)

                lr, l2, epocas, batch = 0.001, 1e-5, 150, 64
                melhor_loss, paciencia, piora = np.inf, 20, 0

                for ep in range(epocas):
                    perm = np.random.permutation(len(X_treino))
                    Xs, ys = X_treino[perm], y_treino[perm]

                    losses = []  # <-- AQUI

                    for i in range(0, len(Xs), batch):
                        Xb, yb = Xs[i:i+batch], ys[i:i+batch]

                        z1 = Xb @ W1.T + b1
                        a1 = np.maximum(0, z1)
                        z2 = a1 @ W2.T + b2
                        a2 = np.maximum(0, z2)
                        logits = a2 @ W3.T + b3

                        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
                        probs = exp / (exp.sum(axis=1, keepdims=True) + 1e-8)

                        loss = -np.mean(np.log(probs[np.arange(len(yb)), yb] + 1e-8))
                        total_loss = loss + 0.5 * l2 * (
                            np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3)
                        )

                        losses.append(total_loss)  # <-- AQUI

                        g3 = probs
                        g3[np.arange(len(yb)), yb] -= 1
                        g3 /= len(yb)

                        g2 = (g3 @ W3) * (z2 > 0)
                        g1 = (g2 @ W2) * (z1 > 0)

                        W3 -= lr * (g3.T @ a2 + l2 * W3)
                        b3 -= lr * g3.sum(axis=0)
                        W2 -= lr * (g2.T @ a1 + l2 * W2)
                        b2 -= lr * g2.sum(axis=0)
                        W1 -= lr * (g1.T @ Xb + l2 * W1)
                        b1 -= lr * g1.sum(axis=0)

                    erro_treino = np.mean(losses)  # <-- AQUI

                    z1v = X_teste @ W1.T + b1
                    a1v = np.maximum(0, z1v)
                    z2v = a1v @ W2.T + b2
                    a2v = np.maximum(0, z2v)
                    logits_val = a2v @ W3.T + b3

                    exp_val = np.exp(logits_val - logits_val.max(axis=1, keepdims=True))
                    probs_val = exp_val / (exp_val.sum(axis=1, keepdims=True) + 1e-8)
                    val_loss = -np.mean(np.log(probs_val[np.arange(len(y_teste)), y_teste] + 1e-8))

                    with open(arquivo_error, "a", encoding="utf-8") as f:
                        f.write(f"{ep};{erro_treino:.8f};{val_loss:.8f}\n")

                    if val_loss < melhor_loss:
                        melhor_loss, piora = val_loss, 0
                        W1b, b1b = W1.copy(), b1.copy()
                        W2b, b2b = W2.copy(), b2.copy()
                        W3b, b3b = W3.copy(), b3.copy()
                    else:
                        piora += 1

                    if piora >= paciencia:
                        W1, b1 = W1b, b1b
                        W2, b2 = W2b, b2b
                        W3, b3 = W3b, b3b
                        break

                logits_final = a2v @ W3.T + b3



            probs_final = np.exp(logits_final - logits_final.max(axis=1, keepdims=True))
            probs_final /= probs_final.sum(axis=1, keepdims=True)
            pred = probs_final.argmax(axis=1)

            acur = (pred == y_teste).mean()
            acuracias.append(acur)

            print(f"Acurácia: {acur:.4f}")

            if acur > melhor_fold[1]:
                melhor_fold = (fold_id, acur)
            if acur < pior_fold[1]:
                pior_fold = (fold_id, acur)

        print(f"\nAcurácia média: {np.mean(acuracias):.4f} ± {np.std(acuracias):.4f}")
        print(f"Melhor fold: {melhor_fold}")
        print(f"Pior fold: {pior_fold}")
