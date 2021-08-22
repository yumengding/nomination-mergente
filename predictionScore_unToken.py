from transformers import FlaubertWithLMHeadModel, FlaubertTokenizer
import torch
from torch.nn import functional as F
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

tokenizer = FlaubertTokenizer.from_pretrained("flaubert/flaubert_base_uncased")
model = FlaubertWithLMHeadModel.from_pretrained("flaubert/flaubert_base_uncased")

handle = open("mask682.txt","r")
handle = handle.readlines()

fichier = open("score de prediction682.txt", "w")
for line in handle:
    line=line.strip()

    coupe = line.split("**")
    
    mot = coupe[0]
    phrase =coupe[1]





    sequence = eval(f"f'''{phrase}'''")

    token_ids = tokenizer.encode(sequence, return_tensors='pt')
    mask_token_index = torch.where(token_ids == tokenizer.mask_token_id)[1]

    token_logits = model(token_ids).logits
    softmax = F.softmax(token_logits, dim=-1)

    mask_token_logits = token_logits[0, mask_token_index, :]
    mask_token_softmax = softmax[0, mask_token_index, :]

    idx = torch.topk(mask_token_logits, 50000, dim=1).indices[0].tolist()
   
    words = []
    for token in idx:
        words.append(tokenizer.decode([token]))
        

    originalTokenId = tokenizer.encode(mot, return_tensors="pt")[0, 1]

    mot_miniscule=mot.lower()
    if mot_miniscule in words:
           
        fichier.write(str(words.index(mot_miniscule) + 1) + "\t" + mot + "\t" + phrase +"\t" +str(mask_token_logits[0, originalTokenId].item())+"\t"+str(mask_token_softmax[0, originalTokenId].item())+"\n")
    else:
		fichier.write("pas prédit" + "\t" + mot + "\t" + phrase +"\t" +str(mask_token_logits[0, originalTokenId].item())+"\t"+str(mask_token_softmax[0, originalTokenId].item())+"\n")



