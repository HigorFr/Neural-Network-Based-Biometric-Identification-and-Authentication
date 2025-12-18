import os
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog, local_binary_pattern
from tqdm import tqdm
import pandas as pd


caminho_root = "Código/Dataset/"


#Lê o identity_CelebA.txt
df = pd.read_csv("Código/Dataset/identity_CelebA.txt", sep=" ", names=["img", "id"])

#Conta quantas imagens cada ID tem
contagem = df["id"].value_counts()

#Seleciona o top 20%
qtd_ids = int(len(contagem) * 0.20)
ids_escolhidos = contagem.head(qtd_ids).index

#Filtra o dataframe
df_filtrado = df[df["id"].isin(ids_escolhidos)]

#Deixei em array do numpy para ficar mais fácil de usar em baixo
imgs_filtradas = df_filtrado["img"].to_numpy()
ids_filtrados = df_filtrado["id"].to_numpy()


print(f"total de classes escolhidas: {len(ids_escolhidos)}")
print(f"total de imagens dessas classes: {len(imgs_filtradas)}")


#Aplicação do HOG

usar_descritor = "HOG"   # HOG ou "LBP"

vetores = []
rótulos = []

caminho_imgs = os.path.join(caminho_root, "Img_align_celeba")

for nome_img, ident in tqdm(zip(imgs_filtradas, ids_filtrados), total=len(imgs_filtradas)):
    caminho = os.path.join(caminho_imgs, nome_img)
    img = imread(caminho)

    #deixa em tom de cinza
    if img.ndim == 3:
        img = rgb2gray(img)

    img = resize(img, (128, 128), anti_aliasing=True, preserve_range=True)

    if usar_descritor == "HOG":
        descritor = hog(
        img,
        orientations=8,
        pixels_per_cell=(16,16),
        cells_per_block=(2,2),
        block_norm='L2-Hys'
        )


    else:  # LBP
        P = 8
        R = 1
        img_uint8 = (img * 255).astype(np.uint8)
        lbp = local_binary_pattern(img_uint8, P, R, method="uniform")
        n_bins = P + 2
        descritor, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(n_bins + 1),
            density=True
        )

    vetores.append(descritor)
    rótulos.append(ident)


vetores = np.array(vetores)
rótulos = np.array(rótulos)

# mapear ids para 0..C-1
ids_unicos = np.unique(rótulos)
mapeamento = {idv: i for i, idv in enumerate(ids_unicos)}
rótulos = np.array([mapeamento[i] for i in rótulos])
num_classes = len(ids_unicos)

print(vetores.shape)
print(np.isnan(vetores).sum())


#Salvar arquivo que a main vai usar

nome_arquivo = f"Código/descritores_{usar_descritor}.npz"

np.savez(
    nome_arquivo,
    vetores=vetores,
    rotulos=rótulos,
    ids_unicos=ids_unicos,
    tipo=usar_descritor
)

print(f"\n Arquivo salvo como: {nome_arquivo}")
