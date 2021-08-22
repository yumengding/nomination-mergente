import torch
from torch.utils.data import TensorDataset, random_split
import math
from transformers import FlaubertForSequenceClassification, FlaubertTokenizer, AdamW
import os
import numpy as np
import wandb
import random
import numpy as np
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from pytorchtools import EarlyStopping
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"


model_id = "flaubert/flaubert_base_uncased"
tokenizer = FlaubertTokenizer.from_pretrained(model_id, do_lower_case=False)

wandb.init(project="BERT")
wandb.watch_called = False

config = wandb.config
config.batch_size = 16
config.epochs = 5
config.seed = 42
output_model = 'model.pth'


random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)


def lirefichier(nomfichier):
	fichier = open(nomfichier,"r")
	fichier = fichier.readlines()
	listeTag = []
	listePhrase = []
	for line in fichier:
		line=line.strip()
		coupe = line.split("\t")
		tag = int(coupe[0])
		mot = coupe[1]
		phrase = coupe[2]
		phrase = phrase.replace("{tokenizer.mask_token}", mot)
		listePhrase.append(phrase)
		listeTag.append(tag)
	return listePhrase, listeTag


listePhraseTrain,listeTagTrain = lirefichier("train_dataset464_10mots.txt")
listePhraseDEV,listeTagDEV = lirefichier("valide_dataset58_10mots.txt")
listePhraseTEST,listeTagTEST = lirefichier("test_dataset59_10mots.txt")

def datatensor(listedePhrase, listedeTag):
	
	input_ids = []
	attention_masks = []
	for sent in listedePhrase:
		encoded_dict = tokenizer.encode_plus(
			sent,  # Sentence to encode.
			add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
			max_length=256,  # Pad & truncate all sentences.
			pad_to_max_length = True,
			return_attention_mask=True,  # Construct attn. masks.
			return_tensors='pt',  # Return pytorch tensors.
		)
		input_ids.append(encoded_dict['input_ids'])
		attention_masks.append(encoded_dict['attention_mask'])

	input_ids = torch.cat(input_ids, dim=0)
	attention_masks = torch.cat(attention_masks, dim=0)
	labels = torch.tensor(listedeTag)
	
	return input_ids, attention_masks, labels

train_input_ids, train_attention_masks, train_labels =datatensor(listePhraseTrain,listeTagTrain)


train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=config.batch_size,
                                          shuffle=True)

dev_input_ids, dev_attention_masks, dev_labels =datatensor(listePhraseDEV,listeTagDEV)
val_dataset = TensorDataset(dev_input_ids, dev_attention_masks, dev_labels)

eval_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                          batch_size=config.batch_size,
                                          shuffle=False)


test_input_ids, test_attention_masks, test_labels =datatensor(listePhraseTEST,listeTagTEST)


test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=config.batch_size,
                                          shuffle=False)

print("data finit")

model = FlaubertForSequenceClassification.from_pretrained(
    model_id, # utiliser la modele de flaubert
    num_labels = 2  # fot binairy classification,two labels for output 二分类任务的输出标签为 2个.
)


optimizer = AdamW(model.parameters(),lr=5e-5)
                #   lr = 1e-5, # args.learning_rate, default 5e-5, - 默认是 5e-5
                #   eps = 1e-8 # args.adam_epsilon,default 1e-8,  - 默认是 1e-8， 是为了防止衰减率分母除到0
                # )






# number of training steps 的数量: [number of batches] x [number of epochs].
total_steps = len(train_loader) * config.epochs

# 设计 learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)




def save(model, optimizer):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, output_model)
    print('The best model has been saved')




print("train commence")


wandb.watch(model, log="all")
for epoch_i in range(0, config.epochs):
    print('Epoch {:} / {:}'.format(epoch_i + 1, config.epochs))

    total_train_loss = 0
    total_train_accuracy = 0
    model.train()

    for step, batch in enumerate(train_loader):

        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]


        # 清空梯度
        model.zero_grad()

        # forward
        # 参考 https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        output = model(b_input_ids,
                             token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels)
        loss = output.loss
        logits = output.logits

        total_train_loss += loss.item()

        # backward 更新 gradients.
        loss.backward()

        # eviter exploding gradient 减去大于1 的梯度，将其设为 1.0, 以防梯度爆炸.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 更新模型参数
        optimizer.step()

        # 更新 learning rate.
        scheduler.step()

        logit = logits.detach().cpu().numpy()
        label_id = b_labels.to('cpu').numpy()
        # accuracy    计算training 句子的准确度.
        total_train_accuracy += flat_accuracy(logit, label_id)

        # loss        计算batches的平均损失.
    avg_train_loss = total_train_loss / len(train_loader)
    
    # accuracy of train    训练集的准确率.
    avg_train_accuracy = total_train_accuracy / len(train_loader)
    print("  accuracy of train训练准确率: {0:.2f}".format(avg_train_accuracy))
    print("  loss of train平均训练损失 loss: {0:.2f}".format(avg_train_loss))
 
    wandb.log({

        "Train Accuracy": avg_train_accuracy,
        "Train Loss": avg_train_loss
    })

    # ========================================
    #               Validation
    # ========================================


    # model dans l'evaluation        设置 model 为valuation 状态，在valuation状态 dropout layers 的dropout rate会不同
    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in eval_loader:
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]

                                         # 在valuation 状态，不更新权值，不改变计算图
        with torch.no_grad():
           output = model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)

        # 计算 validation loss.
        loss = output.loss
        logits = output.logits
        total_eval_loss += loss.item()
        logit = logits.detach().cpu().numpy()
        label_id = b_labels.to('cpu').numpy()

        # accuracy of validation      计算 validation 句子的准确度.
        total_eval_accuracy += flat_accuracy(logit, label_id)

    # accuracy of validation          计算 validation 的准确率.
    avg_val_accuracy = total_eval_accuracy / len(eval_loader)
    print("")
    print("  accuracy of validation测试准确率: {0:.2f}".format(avg_val_accuracy))


    # loss of validation             计算batches的平均损失.
    avg_val_loss = total_eval_loss / len(eval_loader)



    print("  loss of validation 平均测试损失 Loss: {0:.2f}".format(avg_val_loss))
  
    if epoch_i==4:
        save(model, optimizer)
    

    wandb.log({

        "DEV Accuracy": avg_val_accuracy,
        "DEV Loss": avg_val_loss
    })

print("test")
checkpoint = torch.load(output_model, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()


total_test_accuracy = 0


for batch in test_loader:

    b_input_ids = batch[0]
    b_input_mask = batch[1]
    b_labels = batch[2]


    with torch.no_grad():
        output = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)

        # 计算 validation loss.

    logits = output.logits

    logit = logits.detach().cpu().numpy()
    label_id = b_labels.to('cpu').numpy()

        # 计算 test 句子的准确度.
    total_test_accuracy += flat_accuracy(logit, label_id)
    print(np.argmax(logit, axis=1).flatten())
    # accuracy of test           计算 test 的准确率.
avg_test_accuracy = total_test_accuracy / len(test_loader)
print("")
print("  accuracy of test测试准确率: {0:.2f}".format(avg_test_accuracy))
