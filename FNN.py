import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import os
import random
import wandb
from transformers import FlaubertModel, FlaubertTokenizer
from pytorchtools import EarlyStopping
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"



model_id = "flaubert/flaubert_base_uncased"
tokenizer = FlaubertTokenizer.from_pretrained(model_id, do_lower_case=False)
flaubert = FlaubertModel.from_pretrained(model_id, output_hidden_states=True)


wandb.init(project="FNN")
wandb.watch_called = False
config = wandb.config

# les paramères
# dim_input =3072 #con4couches
dim_input = 768
dim_hidden = 100
config.epochs = 100
patience = 20
config.seed = 42


random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)




# construire la modèle
class FNN(nn.Module):
    def __init__(self , dim_input, dim_hidden):
        super(FNN, self).__init__()

        self.fc = nn.Sequential(

        nn.Linear(dim_input, dim_hidden),  # output = M * input   (matrice de taille dim_hidden x dim_input )

        nn.Tanh(),  # output = tanh(input)   (fonction appliquée individuellement pour chaque coefficient)
        nn.Dropout(p=0.75),
        nn.Linear(dim_hidden, 1),  # output = M' * input  (matrice de taille 1 x dim_hidden)

        nn.Sigmoid())  # output = Sigmoid(input)  (fonction appliquée individuellement pour chaque coefficient))

    def forward(self, x):
        out = self.fc(x)
        return out


def tagetphrase(nomFicher):
    fichier=open(nomFicher,"r")
    fichier=fichier.readlines()

    listeTag=[]

    listeVector = []
    i=0
    for line in fichier:
        line=line.strip()
        coupe=line.split("\t")
        tag=coupe[0]
        listeTag.append(tag)

        mot=coupe[1]
        phrase=coupe[2]

        phrase1=eval(f"f'''{phrase}'''")
        token_ids = tokenizer.encode(phrase1, return_tensors='pt')
        mask_token_index = torch.where(token_ids == tokenizer.mask_token_id)[1]
        id_mot = mask_token_index.item()

        ####################################### 4 couches
        # marked_phrase = phrase.replace("{tokenizer.mask_token}",mot)
        # tokenized_phrase = tokenizer.tokenize(marked_phrase)
        # tokenized_phrase.insert(0,"<s>")
        # tokenized_phrase.append("</s>")
        # indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_phrase)
        #
        #
        # segments_ids = [1] * len(tokenized_phrase)
        #
        # tokens_tensor = torch.tensor([indexed_tokens])
        # segments_tensors = torch.tensor([segments_ids])
        #
        #
        # flaubert.eval()
        #
        # with torch.no_grad():  # 将输入传入模型，得到每一层的输出信息，这里的encoded_layers为12层，可以打印验证
        #     outputs = flaubert(tokens_tensor, segments_tensors)
        #     hidden_states = outputs[1]


        #sum of last four layer
        # word_embed_sum = torch.stack(hidden_states[-4:]).sum(0)
        # word_embed_sum=word_embed_sum.squeeze()
        # # print(word_embed_sum.size())
        # # print(word_embed_sum[id_mot-1])
        # contextualizedVec = word_embed_sum[id_mot]

        # concatenate last four layers
        # word_embed_concatenate = torch.cat([hidden_states[i] for i in [-1, -2, -3, -4]], dim=-1)
        # word_embed_concatenate = word_embed_concatenate.squeeze()
        #
        # contextualizedVec = word_embed_concatenate[id_mot]
        ##################################################################################
        #####################################1 couche
        marked_phrase = phrase.replace("{tokenizer.mask_token}", mot)

        # obtenir les vecteurs de phrase
        tokenIds = tokenizer.encode(phrase)
        inputVector = torch.tensor([tokenIds])
        flaubert.eval()
        with torch.no_grad():
            outputVectors = flaubert(inputVector)[0]
        outputVectors = outputVectors.squeeze(0)


        # obtenir le vecteur du mot masqué
        contextualizedVec = outputVectors[id_mot]

################################################################################################
        listeVector.append(contextualizedVec)
        i = i + 1
        print(i)





    return listeTag,listeVector


# lire les entrées

liste_train_tag,liste_train_vector = tagetphrase("train_dataset464_10mots.txt")
liste_eval_tag, liste_eval_vector= tagetphrase("valide_dataset58_10mots.txt")
liste_test_tag, liste_test_vector = tagetphrase("test_dataset59_10mots.txt")
print("data finit")

# appelle la modèle
model = FNN(dim_input, dim_hidden)
loss_function = nn.BCELoss()  # Creates a criterion that measures the Binary Cross Entropy between the target and the output
optimizer = optim.Adam(model.parameters(),lr=0.0001)


###############################################################################################################
# commence l'entraînement
print("train")


input_vectors_train = liste_train_vector
output_bool_train = [int(x) for x in liste_train_tag]
output_tensor_train = [torch.tensor([b], dtype=torch.float) for b in output_bool_train]


early_stopping = EarlyStopping(patience=patience, verbose= True)
wandb.watch(model, log="all")

for epoch in range(config.epochs):  # itération sur les données d'entraînement
    model.train()
    loss_epoch = 0
    cpt_juste=0

# shuffle la liste de l'entrainement
    state = np.random.get_state()
    np.random.shuffle(input_vectors_train)
    np.random.set_state(state)
    np.random.shuffle(output_tensor_train)
    
    for input_vector, target in zip(input_vectors_train, output_tensor_train):
        optimizer.zero_grad()

        # calcul de l'output du modèle
        output = model(input_vector)

        loss = loss_function(output, target)
        prediction = output > 0.5  # -> permet de calculer l'exactitude du modèle sur les données de test
        # ~ if int(prediction) != int(target):#si la prédiction n'est pas la même que la cible
        # ~ print(prediction, int(target))
        if int(prediction) == int(target):
            cpt_juste += 1

        loss_epoch += loss.detach().numpy()

        loss.backward(retain_graph=True)  # il y a toujours une erreur sans retain_graph=True pour la fonction backward

        optimizer.step()

    accuracy = 100 * cpt_juste / len(output_tensor_train)
    print(f"Epoch {epoch} loss = {loss_epoch} précision sur l'epoch {accuracy}" + " entrainement")

    wandb.log({

        "Train Accuracy": accuracy,
        "Train Loss": loss_epoch
    })



###### commence l'evaluation dans chaque epoch
    print("evaluation")

    input_vectors_eval = liste_eval_vector  # liste de tenseurs issus du train set
    # output sous la forme d'une liste de booleen
    output_bool_eval = [int(x) for x in liste_eval_tag]
    # output sous la forme de tensors à 1 élément (il faut des float)
    output_tensor_eval = [torch.tensor([b], dtype=torch.float) for b in output_bool_eval]

    model.eval()
    cpt_juste_eval = 0
    loss_eval = 0
    with torch.no_grad():
        for input_vector, target in zip(input_vectors_eval, output_tensor_eval):
            output = model(input_vector)

            loss = loss_function(output, target)
            loss_eval += loss.detach().numpy()
            # l'output est entre 0 et 1 et s'interprète comme P(compositionnel | exemple)
            # Pour extraire une prédiction, on considère que la réponse du modèle est 1 si output > 0.5 et 0 sinon
            prediction = output > 0.5  # -> permet de calculer l'exactitude du modèle sur les données de test
            # ~ if int(prediction) != int(target):#si la prédiction n'est pas la même que la cible
            # ~ print(prediction, int(target))
            if int(prediction) == int(target):
                cpt_juste_eval += 1
    accuracy = 100 * cpt_juste_eval / len(output_tensor_eval)
    validation_time = format_time(time.time() - t0)
    print(f" evaluation : loss = {loss_eval} précision  {accuracy}")

    early_stopping(loss_eval, model)
    #  early stopping
    if early_stopping.early_stop:
        print("Early stopping")

        break

    wandb.log({

        "DEV Accuracy": accuracy,
        "DEV Loss": loss_eval
    })

print("save model")
print("\n\n")

###############################################################################################

print("\n\n")

###############################################################################################################
#commence le test
print("test")

model.load_state_dict(torch.load('checkpoint.pt'))
model.eval()


cpt_juste_test = 0
input_vectors = liste_test_vector  # liste de tenseurs issus du train set
output_bool = [int(x) for x in liste_test_tag]
output_tensor = [torch.tensor([b], dtype=torch.float) for b in output_bool]

with torch.no_grad():
    for input_vector, target in zip(input_vectors, output_tensor):
        output = model(input_vector)


            # l'output est entre 0 et 1 et s'interprète comme P(compositionnel | exemple)
            # Pour extraire une prédiction, on considère que la réponse du modèle est 1 si output > 0.5 et 0 sinon
        prediction = output > 0.5  # -> permet de calculer l'exactitude du modèle sur les données de test
            # ~ if int(prediction) != int(target):#si la prédiction n'est pas la même que la cible
            # ~ print(prediction, int(target))
        if int(prediction) == int(target):
            cpt_juste_test += 1
        print(int(prediction))
print(cpt_juste_test)
accuracy = 100 * cpt_juste_test / len(output_tensor)
print(f" précision  {accuracy}")
