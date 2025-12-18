import numpy as np
from sklearn.model_selection import KFold
import os
import datetime


#funções auxiliares 
def gerar_pares(vetores, rotulos, pares_positivos=12, pares_negativos=12): #isso aqui é para gerar os pares para o modelo tentar adivionhar se são ou não a mesma pessoa
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


#aqui é para inicializar os pesos com 
def inicializar_pesos_final(inp, out):
    return np.random.randn(out, inp) * np.sqrt(2.0 / inp)

def inicializar_pesos_xavier(inp, out):
    return np.random.randn(out, inp) * np.sqrt(2.0 / (inp + out))



#definindo os parametros gerais
timestamp = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
descritores = ["HOG","LBP"]  #Isso é para fazer o for e deixar o pc rodando sem intervenção (Antes era um parametro colocado manualmente)
modelos = ["linear", "mlp"]
random_state = 42



for descritor in descritores:

    #Carrega os dados e separa cada parte
    data = np.load(f"Código/descritores_{descritor}.npz")
    vetores = data["vetores"]
    rotulos = data["rotulos"]
    ids_unicos = data["ids_unicos"]
    tipo = data["tipo"]

    print("Arquivo carregado:", tipo)
    print("Número real de IDs:", len(ids_unicos))#20% já filtrado pelo extrator HOG e LBP


    X_pairs, y_pairs = gerar_pares(vetores, rotulos) #Gera os pares de fato
    print("Total de pares:", len(X_pairs)) 
    print("Dimensão dos pares:", X_pairs.shape)

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state) #Faz o fold  (Com k= conforme pedido)

    
    for modelo in modelos:


        print(f"MODELO: {modelo.upper()}")
        print(f"------------------------------------------------")

        #ISso aqui é só para configurar onde ele vai guardar as saídas
        pasta_base = f"Resultados_Autenticacao/{descritor}/{modelo}"
        os.makedirs(pasta_base, exist_ok=True)
        arquivo_config = os.path.join(pasta_base, "run_config.txt")
        arquivo_error = os.path.join(pasta_base, "run_error.txt")

        #já marca inicio da execção
        with open(arquivo_config, "w", encoding="utf-8") as f:
            f.write(f"Execução: {timestamp}\nDescritor: {descritor}\nModelo: {modelo}\n")

        with open(arquivo_error, "w", encoding="utf-8") as f:
            f.write(f"Execucao em {timestamp}\n")

        #placeholder que vão ser inseridos durante a execução (Só vai servir pra colocar no log)
        acuracias = []
        melhor_fold = (-1, -np.inf)
        pior_fold = (-1, np.inf)

        #Aqui aplica o fold
        for fold_id, (treino_idx, teste_idx) in enumerate(kf.split(X_pairs)):
            print(f"\n--- Fold {fold_id} ---") #Printar só pra facilitar na hora de ver

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

            #Aqui começa o linear
            if modelo == "linear":
                W = inicializar_pesos_xavier(n_atrib, num_classes)
                b = np.zeros(num_classes)

                #Parametros de fato:
                lr = 0.005 #taxa de aprendizado
                l2 = 1e-5 #Penalidade (baixa)
                epocas = 100 #auto-explicativo
                batch = 128  #auto-explicativo
                paciencia = 20 #tempo sem melhorar

                #só placeholders para salvar informação
                melhor_loss = np.inf
                piora = 0

                for ep in range(epocas):

                    losses = []   #Aramzena os losses para tirar a média e colocar no log

                    perm = np.random.permutation(len(X_treino))
                    X_treino, y_treino = X_treino[perm], y_treino[perm]


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
                h1, h2 = 64, 16  #arquitura, aqui defini 64 pra primeira camada e 16 na outra, porque meu pc não aguenta muito

                #define os pesos com bases naquelas funções auxiliares
                W1, b1 = inicializar_pesos_final(n_atrib, h1), np.zeros(h1)
                W2, b2 = inicializar_pesos_final(h1, h2), np.zeros(h2)
                W3, b3 = inicializar_pesos_xavier(h2, num_classes), np.zeros(num_classes)

               #Parametros de fato:
                lr = 0.005 #taxa de aprendizado
                l2 = 1e-5 #Penalidade (baixa)
                epocas = 100 #auto-explicativo
                batch = 128  #auto-explicativo
                paciencia = 20 #tempo sem melhorar


                melhor_loss = np.inf
                piora = 0

                for ep in range(epocas):
                    perm = np.random.permutation(len(X_treino))
                    Xs, ys = X_treino[perm], y_treino[perm]

                    losses = [] 

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
                        #vai marcando se nõo ta melhorando
                        piora += 1
                    
                    #Aqui dá earling stop se não melhorar
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

            #só para registrar qual é o melhor fold
            if acur > melhor_fold[1]:
                melhor_fold = (fold_id, acur)
            if acur < pior_fold[1]:
                pior_fold = (fold_id, acur)


        #Colocando no arquivo de config os parametros (Em ingles, conforme estava no pptx da atividade)
        with open(arquivo_config, "w", encoding="utf-8") as f:
            f.write(f"EXECUTION_TIMESTAMP: {timestamp}\n")
            f.write(f"DESCRIPTOR: {descritor}\n")
            f.write(f"MODEL: {modelo}\n\n")

        
            if modelo == "linear":
                f.write("LINEAR_SPECIFICATION: ('input_layer', {}, 'softmax', 'cross_entropy')\n".format(n_atrib))
                f.write(f"LINEAR_OPERATION_LR: {lr}\n")
                f.write(f"LINEAR_OPERATION_L2: {l2}\n")
                f.write(f"LINEAR_OPERATION_MAX_EPOCHS: {epocas}\n")
                f.write(f"LINEAR_OPERATION_BATCH_SIZE: {batch}\n")
                f.write(f"LINEAR_OPERATION_PATIENCE: {paciencia}\n")
            else:
                f.write("MLP_SPECIFICATION: ('layer 0', {}, 'relu', 'cross_entropy')\n".format(h1))
                f.write("MLP_SPECIFICATION: ('layer 1', {}, 'relu', 'cross_entropy')\n".format(h2))
                f.write("MLP_SPECIFICATION: ('output_layer', {}, 'softmax', 'cross_entropy')\n".format(num_classes))
                f.write(f"MLP_OPERATION_LR_METHOD: FIX\n")
                f.write(f"MLP_OPERATION_LR_PARAMS: {lr}\n")
                f.write(f"MLP_OPERATION_INITIALISATION: He_2015\n")
                f.write(f"MLP_OPERATION_MAX_EPOCHS: {epocas}\n")
                f.write(f"MLP_OPERATION_MIN_EPOCHS: 1\n")
                f.write(f"MLP_OPERATION_STOP_WINDOW: {paciencia}\n")
                f.write(f"MLP_OPERATION_BATCH_SIZE: {batch}\n")
                f.write(f"MLP_OPERATION_L2: {l2}\n")

