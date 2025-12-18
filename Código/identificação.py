import numpy as np
from sklearn.model_selection import KFold
import os
import datetime

timestamp = datetime.datetime.now().strftime("%d/%m/%Y %H:%M") #vai ser usado na hora de fazer o txt


#Carrega os npz que o extrator HOG/LBP.py gerou

descritor = "HOG" #Colocar HOG OU LBP

data = np.load(f"Código/descritores_{descritor}.npz")  # ou LBP

vetores = data["vetores"]
rotulos = data["rotulos"]
ids_unicos = data["ids_unicos"]
tipo = data["tipo"]

print("Arquivo carregado:", tipo)



#Aqui começa o modelo de fato:

modelo = "mlp"   #Aqui eu seleciono se será "linear" ou "mlp"
pasta_base = f"Resultados_Identificação/{descritor}/{modelo}"
os.makedirs(pasta_base, exist_ok=True)


#Aqui é só para configurar os caminhos automaicamente com base no pptx no edisciplinas
arquivo_config = os.path.join(pasta_base, "run_config.txt")
arquivo_error = os.path.join(pasta_base, "run_error.txt")
arquivo_model = os.path.join(pasta_base, "model.dat")

with open(arquivo_config, "w") as f:
    f.write(f"Execucao em {timestamp}\n")

with open(arquivo_error, "w") as f:
    f.write(f"Execucao em {timestamp}\n")


kf = KFold(n_splits=5, shuffle=True, random_state=42)
acuracias = []

melhor_fold = (-1, -np.inf)
pior_fold   = (-1,  np.inf)


for fold_id, (treino_idx, teste_idx) in enumerate(kf.split(vetores)):
    X_treino = vetores[treino_idx]
    y_treino = rotulos[treino_idx]
    X_teste = vetores[teste_idx]
    y_teste = rotulos[teste_idx]

    # Normalizar
    media = X_treino.mean(axis=0)
    desvio = X_treino.std(axis=0) + 1e-6

    X_treino = (X_treino - media) / desvio
    X_teste = (X_teste - media) / desvio

    n_atrib = X_treino.shape[1]

    num_classes = len(ids_unicos)

    historico_epocas = []

    if modelo == "linear":
        # inicialização
        W = np.random.randn(num_classes, n_atrib) * 0.01
        b = np.zeros(num_classes)

        #PARAMETROs do modelo linear
        taxa = 0.01
        épocas = 15
        tam_batch = 64

        for ep in range(épocas):
            perm = np.random.permutation(len(X_treino))
            X_treino = X_treino[perm]
            y_treino = y_treino[perm]


            for i in range(0, len(X_treino), tam_batch):
                Xb = X_treino[i:i+tam_batch]
                yb = y_treino[i:i+tam_batch]

                logits = Xb @ W.T + b
                e = np.exp(logits - logits.max(axis=1, keepdims=True))
                probs = e / e.sum(axis=1, keepdims=True)

                loss = -np.log(probs[np.arange(len(yb)), yb]).mean()



                grad_logits = probs
                grad_logits[np.arange(len(yb)), yb] -= 1
                grad_logits /= len(yb)

                grad_W = grad_logits.T @ Xb
                grad_b = grad_logits.sum(axis=0)

                W -= taxa * grad_W
                b -= taxa * grad_b

        #avaluar
            logits_val = X_teste @ W.T + b
            e_val = np.exp(logits_val - logits_val.max(axis=1, keepdims=True))
            probs_val = e_val / e_val.sum(axis=1, keepdims=True)

            loss_val = -np.log(probs_val[np.arange(len(y_teste)), y_teste]).mean()

            historico_epocas.append((ep, loss, loss_val))










    else:  #caso for MLP
        H = 128
        W1 = np.random.randn(H, n_atrib) * 0.01
        b1 = np.zeros(H)
        W2 = np.random.randn(num_classes, H) * 0.01
        b2 = np.zeros(num_classes)

        #PARAMETROs do MLP
        taxa = 0.001
        épocas = 200
        pac = 64
        paciência = 3
        piora = 0
        melhor_valor = 0

        for ep in range(épocas):
            perm = np.random.permutation(len(X_treino))
            X_treino = X_treino[perm]
            y_treino = y_treino[perm]

            total_loss_treino = 0
            total_batches = 0

            for i in range(0, len(X_treino), pac):
                Xb = X_treino[i:i+pac]
                yb = y_treino[i:i+pac]

                #forward
                h = Xb @ W1.T + b1
                h_relu = np.maximum(0, h)
                logits = h_relu @ W2.T + b2
                e = np.exp(logits - logits.max(axis=1, keepdims=True))
                probs = e / e.sum(axis=1, keepdims=True)

                #backprop
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


            #loss = -np.log(probs[np.arange(len(yb)), yb]).mean() tava dando erro de algum 0 cair no log
            loss = -np.log(np.clip(probs[np.arange(len(yb)), yb], 1e-12, 1)).mean()
            total_loss_treino += loss
            total_batches += 1


            #avaliação
            h_t = X_teste @ W1.T + b1
            h_t = np.maximum(0, h_t)
            logits_t = h_t @ W2.T + b2
            
            e_val = np.exp(logits_t - logits_t.max(axis=1, keepdims=True))
            probs_val = e_val / e_val.sum(axis=1, keepdims=True)
            
            val_loss = -np.log(probs_val[np.arange(len(y_teste)), y_teste]).mean()
            acur = (logits_t.argmax(axis=1) == y_teste).mean()

            historico_epocas.append((ep, loss, val_loss))


            #Aqui é para parar as épocas caso ele não melhorou muito
            if acur > melhor_valor:
                melhor_valor = acur
                piora = 0
            else:
                piora += 1
                if piora >= paciência:
                    break

    pred = probs_val.argmax(axis=1)
    acur = (pred == y_teste).mean()
    acuracias.append(acur)

    # atualizar melhor e pior

    if acur > melhor_fold[1]:
        melhor_fold = (fold_id, acur)
    if acur < pior_fold[1]:
        pior_fold = (fold_id, acur)


with open(arquivo_error, "a") as f:
    f.write("\n")
    for ep, loss_treino, loss_val in historico_epocas:
        f.write(f"{ep};{loss_treino:.10f};{loss_val:.10f}\n")

with open(arquivo_config, "a") as f:
    f.write(f"\nDescritor: {descritor}\n")
    f.write(f"Modelo: {modelo}\n")
    f.write(f"Acuracias: {acuracias}\n")
    f.write(f"Acuracia media: {np.mean(acuracias):.4f}\n")
    f.write(f"Melhor fold: {melhor_fold}\n")
    f.write(f"Pior fold: {pior_fold}\n")


print("Acurácias por fold:", acuracias)
print("Acurácia média:", np.mean(acuracias))

