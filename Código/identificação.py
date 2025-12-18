import numpy as np
from sklearn.model_selection import KFold
import os
import datetime
from sklearn.preprocessing import LabelEncoder


#funções auxiliares para inicializar pesos
def inicializar_weights_he(inp, out): #He initialization
    return np.random.randn(out, inp) * np.sqrt(2.0 / inp)

def inicializar_weights_xavier(inp, out): #Xavier/Glorot initialization
    return np.random.randn(out, inp) * np.sqrt(2.0 / (inp + out))



#configurações gerais
timestamp = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
descritores = ["HOG", "LBP"]  #descritores a processar
modelos = ["linear", "mlp"]    #tipos de modelo
random_state = 42               #seed para reproducibilidade

for descritor in descritores:

    #carregar dados
    data = np.load(f"Código/descritores_{descritor}.npz")
    vetores = data["vetores"]
    rotulos = data["rotulos"]
    ids_unicos = data["ids_unicos"]
    tipo = data["tipo"]

    print("Arquivo carregado:", tipo)
    print(f"Número de classes (IDs): {len(ids_unicos)}")
    print(f"Número de amostras: {len(vetores)}")

    #codificar rótulos
    le = LabelEncoder()  #transforma rótulos em números
    rotulos_encoded = le.fit_transform(rotulos)
    num_classes = len(le.classes_)

    print(f"Classes codificadas: {num_classes}")

    #configura K-Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    for modelo in modelos:

        print(f"\n{'='*60}")
        print(f"MODELO: {modelo.upper()}")
        print(f"{'='*60}")

        #cria pasta para salvar resultados
        pasta_base = f"Resultados_Identificacao/{descritor}/{modelo}"
        os.makedirs(pasta_base, exist_ok=True)

        arquivo_config = os.path.join(pasta_base, "run_config.txt")
        arquivo_error = os.path.join(pasta_base, "run_error.txt")

        #inicia arquivos de log
        with open(arquivo_config, "w", encoding="utf-8") as f:
            f.write(f"Execução: {timestamp}\nDescritor: {descritor}\nModelo: {modelo}\n")

        with open(arquivo_error, "w", encoding="utf-8") as f:
            f.write(f"Execucao em {timestamp}\n")

        #listas para métricas
        acuracias = []
        precisoes = []
        recalls = []
        f1_scores = []

        melhor_fold = (-1, -np.inf)  #melhor fold
        pior_fold = (-1, np.inf)     #pior fold

        #loop K-Fold
        for fold_id, (treino_idx, teste_idx) in enumerate(kf.split(vetores)):
            print(f"\n--- Fold {fold_id} ---")

            #separa treino e teste
            X_treino = vetores[treino_idx]
            y_treino = rotulos_encoded[treino_idx]
            X_teste = vetores[teste_idx]
            y_teste = rotulos_encoded[teste_idx]

            #análise de balanceamento
            unique_train, counts_train = np.unique(y_treino, return_counts=True)
            unique_test, counts_test = np.unique(y_teste, return_counts=True)

            print(f"Amostras treino: {len(X_treino)} (mín: {counts_train.min()}, máx: {counts_train.max()})")
            print(f"Amostras teste: {len(X_teste)} (mín: {counts_test.min()}, máx: {counts_test.max()})")

            #normalização Z-score
            media = X_treino.mean(axis=0)
            desvio = X_treino.std(axis=0) + 1e-8
            X_treino = (X_treino - media) / desvio
            X_teste = (X_teste - media) / desvio

            n_atrib = X_treino.shape[1]  #número de atributos

            historico_epocas = []  #para salvar losses por época

            #MODELO LINEAR
            if modelo == "linear":
                W = inicializar_weights_xavier(n_atrib, num_classes)  #pesos
                b = np.zeros(num_classes)                              #bias

                #hiperparâmetros
                lr = 0.005
                l2 = 1e-4
                epocas = 20
                batch = 64
                melhor_val_loss = np.inf
                paciencia = 5
                piora = 0

                for ep in range(epocas):
                    #embaralha os dados
                    perm = np.random.permutation(len(X_treino))
                    X_treino_shuffled = X_treino[perm]
                    y_treino_shuffled = y_treino[perm]

                    losses_batch = []  #armazena losses do batch

                    for i in range(0, len(X_treino_shuffled), batch):
                        Xb = X_treino_shuffled[i:i+batch]
                        yb = y_treino_shuffled[i:i+batch]

                        #forward
                        logits = Xb @ W.T + b
                        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                        probs = exp / (exp.sum(axis=1, keepdims=True) + 1e-8)

                        #loss com L2
                        loss = -np.mean(np.log(probs[np.arange(len(yb)), yb] + 1e-8))
                        total_loss = loss + 0.5 * l2 * np.sum(W * W)
                        losses_batch.append(total_loss)

                        #backward
                        grad = probs.copy()
                        grad[np.arange(len(yb)), yb] -= 1
                        grad /= len(yb)
                        W -= lr * (grad.T @ Xb + l2 * W)
                        b -= lr * grad.sum(axis=0)

                    erro_treino = np.mean(losses_batch)

                    #validação
                    logits_val = X_teste @ W.T + b
                    exp_val = np.exp(logits_val - np.max(logits_val, axis=1, keepdims=True))
                    probs_val = exp_val / (exp_val.sum(axis=1, keepdims=True) + 1e-8)
                    val_loss = -np.mean(np.log(probs_val[np.arange(len(y_teste)), y_teste] + 1e-8))

                    #early stopping
                    if val_loss < melhor_val_loss:
                        melhor_val_loss, piora = val_loss, 0
                        W_best, b_best = W.copy(), b.copy()
                    else:
                        piora += 1

                    #log do batch
                    with open(arquivo_error, "a", encoding="utf-8") as f:
                        f.write(f"{ep};{erro_treino:.8f};{val_loss:.8f}\n")

                    historico_epocas.append((ep, erro_treino, val_loss))

                    if piora >= paciencia:
                        W, b = W_best, b_best
                        break

                #avaliação final
                logits_final = X_teste @ W.T + b
                exp_final = np.exp(logits_final - np.max(logits_final, axis=1, keepdims=True))
                probs_final = exp_final / (exp_final.sum(axis=1, keepdims=True) + 1e-8)


            #MODELO MLP
            else:
                #arquitetura: input -> hidden1 -> hidden2 -> output
                h1, h2 = 64, 16
                W1, b1 = inicializar_weights_he(n_atrib, h1), np.zeros(h1)
                W2, b2 = inicializar_weights_he(h1, h2), np.zeros(h2)
                W3, b3 = inicializar_weights_xavier(h2, num_classes), np.zeros(num_classes)

                #hiperparâmetros
                lr = 0.001
                l2 = 1e-5
                epocas = 20
                batch = 64
                dropout_rate = 0.5
                melhor_val_loss = np.inf
                melhor_val_acc = 0
                paciencia = 5
                piora = 0

                for ep in range(epocas):
                    #shuffle
                    perm = np.random.permutation(len(X_treino))
                    Xs, ys = X_treino[perm], y_treino[perm]

                    losses_batch = []

                    for i in range(0, len(Xs), batch):
                        Xb, yb = Xs[i:i+batch], ys[i:i+batch]

                        #forward com dropout
                        z1 = Xb @ W1.T + b1
                        a1 = np.maximum(0, z1)
                        if dropout_rate > 0:
                            mask1 = np.random.binomial(1, 1-dropout_rate, size=a1.shape) / (1-dropout_rate)
                            a1 *= mask1

                        z2 = a1 @ W2.T + b2
                        a2 = np.maximum(0, z2)
                        if dropout_rate > 0:
                            mask2 = np.random.binomial(1, 1-dropout_rate, size=a2.shape) / (1-dropout_rate)
                            a2 *= mask2

                        logits = a2 @ W3.T + b3
                        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                        probs = exp / (exp.sum(axis=1, keepdims=True) + 1e-8)

                        #loss com L2
                        loss = -np.mean(np.log(probs[np.arange(len(yb)), yb] + 1e-8))
                        total_loss = loss + 0.5 * l2 * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))
                        losses_batch.append(total_loss)

                        #backward
                        g3 = probs.copy()
                        g3[np.arange(len(yb)), yb] -= 1
                        g3 /= len(yb)
                        g2 = (g3 @ W3) * (z2 > 0)
                        if dropout_rate > 0:
                            g2 *= mask2
                        g1 = (g2 @ W2) * (z1 > 0)
                        if dropout_rate > 0:
                            g1 *= mask1

                        #atualiza pesos
                        W3 -= lr * (g3.T @ a2 + l2 * W3)
                        b3 -= lr * g3.sum(axis=0)
                        W2 -= lr * (g2.T @ a1 + l2 * W2)
                        b2 -= lr * g2.sum(axis=0)
                        W1 -= lr * (g1.T @ Xb + l2 * W1)
                        b1 -= lr * g1.sum(axis=0)

                    erro_treino = np.mean(losses_batch)

                    #validação sem dropout
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

                    #early stopping baseado em acurácia
                    if val_acc > melhor_val_acc:
                        melhor_val_acc = val_acc
                        melhor_val_loss = val_loss
                        piora = 0
                        W1_best, b1_best = W1.copy(), b1.copy()
                        W2_best, b2_best = W2.copy(), b2.copy()
                        W3_best, b3_best = W3.copy(), b3.copy()
                    else:
                        piora += 1

                    #log do batch
                    with open(arquivo_error, "a", encoding="utf-8") as f:
                        f.write(f"{ep};{erro_treino:.8f};{val_loss:.8f}\n")

                    historico_epocas.append((ep, erro_treino, val_loss))

                    if piora >= paciencia:
                        W1, b1 = W1_best, b1_best
                        W2, b2 = W2_best, b2_best
                        W3, b3 = W3_best, b3_best
                        break

                #avaliação final
                z1_final = X_teste @ W1.T + b1
                a1_final = np.maximum(0, z1_final)
                z2_final = a1_final @ W2.T + b2
                a2_final = np.maximum(0, z2_final)
                logits_final = a2_final @ W3.T + b3

                exp_final = np.exp(logits_final - np.max(logits_final, axis=1, keepdims=True))
                probs_final = exp_final / (exp_final.sum(axis=1, keepdims=True) + 1e-8)


            #avaliação do fold
            pred = probs_final.argmax(axis=1)
            acur = (pred == y_teste).mean()
            acuracias.append(acur)

            from sklearn.metrics import precision_score, recall_score, f1_score
            precisao = precision_score(y_teste, pred, average='weighted', zero_division=0)
            recall = recall_score(y_teste, pred, average='weighted', zero_division=0)
            f1 = f1_score(y_teste, pred, average='weighted', zero_division=0)

            precisoes.append(precisao)
            recalls.append(recall)
            f1_scores.append(f1)

            print(f"Acurácia: {acur:.4f}")
            print(f"Precisão: {precisao:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")

            #melhor e pior fold
            if acur > melhor_fold[1]:
                melhor_fold = (fold_id, acur)
            if acur < pior_fold[1]:
                pior_fold = (fold_id, acur)


        #resultados finais do modelo
        print(f"\nAcurácia média: {np.mean(acuracias):.4f} ± {np.std(acuracias):.4f}")
        print(f"Melhor fold: {melhor_fold}")
        print(f"Pior fold: {pior_fold}")

        #salva configuração final
        with open(arquivo_config, "w", encoding="utf-8") as f:
            f.write(f"EXECUTION_TIMESTAMP: {timestamp}\n")
            f.write(f"DESCRIPTOR: {descritor}\n")
            f.write(f"MODEL: {modelo}\n\n")

            if modelo == "linear":
                f.write(f"LINEAR_SPECIFICATION: ('input_layer', {n_atrib}, 'softmax', 'cross_entropy')\n")
                f.write(f"LINEAR_OPERATION_LR_METHOD: FIX\n")
                f.write(f"LINEAR_OPERATION_LR_PARAMS: {lr}\n")
                f.write(f"LINEAR_OPERATION_INITIALISATION: Glorot_Bengio_2010\n")
                f.write(f"LINEAR_OPERATION_MAX_EPOCHS: {epocas}\n")
                f.write(f"LINEAR_OPERATION_BATCH_SIZE: {batch}\n")
                f.write(f"LINEAR_OPERATION_PATIENCE: {paciencia}\n")
                f.write(f"LINEAR_OPERATION_L2: {l2}\n")
            else:
                f.write(f"MLP_SPECIFICATION: ('layer 0', {h1}, 'relu', 'cross_entropy')\n")
                f.write(f"MLP_SPECIFICATION: ('layer 1', {h2}, 'relu', 'cross_entropy')\n")
                f.write(f"MLP_SPECIFICATION: ('output_layer', {num_classes}, 'softmax', 'cross_entropy')\n")
                f.write(f"MLP_OPERATION_LR_METHOD: FIX\n")
                f.write(f"MLP_OPERATION_LR_PARAMS: {lr}\n")
                f.write(f"MLP_OPERATION_INITIALISATION: He_2015\n")
                f.write(f"MLP_OPERATION_MAX_EPOCHS: {epocas}\n")
                f.write(f"MLP_OPERATION_MIN_EPOCHS: 1\n")
                f.write(f"MLP_OPERATION_STOP_WINDOW: {paciencia}\n")
                f.write(f"MLP_OPERATION_BATCH_SIZE: {batch}\n")
                f.write(f"MLP_OPERATION_L2: {l2}\n")
                f.write(f"MLP_OPERATION_DROPOUT_RATE: {dropout_rate}\n")
