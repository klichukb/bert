# coding: utf-8
"""
Author : Dhanush S R
Dataset Used: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
"""

#-------------------------------------------------------------------------------
#--------------- Importing Necessary Libraries ---------------------------------
#-------------------------------------------------------------------------------


import pandas as pd
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, random_split
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import time
import datetime
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from dataloader import get_dataset
from utils import format_time
from model import MultiTaskClassifier

#-------------------------------------------------------------------------------
#----------------------- Preparing Environment ---------------------------------
#-------------------------------------------------------------------------------

if False and torch.backends.mps.is_available():
    # torch.cuda.set_device(0)
    device = torch.device('mps')
    print('Using GPU: ', device)
else:
    device = torch.device('cpu')

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

#-------------------------------------------------------------------------------
#--------------------------- Preparing Dataset ---------------------------------
#-------------------------------------------------------------------------------


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
train_dataset, val_dataset = get_dataset(
        pd.read_csv('./data/dataset.csv'),
    tokenizer = tokenizer,
)

batch_size = 8
train_dataloader = DataLoader(
    train_dataset,
    batch_size = batch_size,
    sampler = RandomSampler(train_dataset)
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size = batch_size,
    sampler = SequentialSampler(val_dataset)
)

print('Data Ready!!')

#-------------------------------------------------------------------------------
#----------------------------- Preparing Model ---------------------------------
#-------------------------------------------------------------------------------


model = MultiTaskClassifier(100, 6).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
criterion = nn.BCELoss()

#-------------------------------------------------------------------------------
#--------------------- Training and Validation ---------------------------------
#-------------------------------------------------------------------------------



epochs = 5
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps = 0,
                                           num_training_steps = total_steps)


training_stats = []
total_t0 = time.time()

best_val_loss = 1e8
true_labels = val_dataset[:][2].numpy()

for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()

    total_train_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):

        if step % 1000 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}. Loss: {:.5f}'.format(step, len(train_dataloader), elapsed, total_train_loss/step))

        b_in_T            = batch[0].to(device)
        b_in_T_attn_masks = batch[1].to(device)
        b_labels          = batch[2].to(device)
        
        model.zero_grad()

        logits = model(b_in_T, b_in_T_attn_masks)
        loss = criterion(logits, b_labels)

        total_train_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    pred_labels = np.empty((0,6))

    # Evaluate data for one epoch
    for batch in val_dataloader:
        
        b_in_T            = batch[0].to(device)
        b_in_T_attn_masks = batch[1].to(device)
        b_labels          = batch[2].to(device)

        with torch.no_grad():
            logits = model(b_in_T, b_in_T_attn_masks)
            loss = criterion(logits, b_labels)

        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        pred_labels = np.concatenate((pred_labels, logits), axis=0)


    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(val_dataloader)

    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    pred_labels = np.array([[int(x >= 0.25) for x in pred_labels[:,i]] for i  in range(6)]).transpose()

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

#     Report the final accuracy, f1-score for this validation run.
    for i in range(6):
        print("  Accuracy: {0:.2f}".format(accuracy_score(true_labels[:,i], pred_labels[:,i])))

    for i in range(6):
        print("  Macro F1-score: {0:.2f}".format(f1_score(true_labels[:,i], pred_labels[:,i], average='macro')))

    for i in range(6):
        print("  Weighted F1-score: {0:.2f}".format(f1_score(true_labels[:,i], pred_labels[:,i], average='weighted')))

    print('Classification Report:')
    for i in range(6):
        print(classification_report(true_labels[:,i], pred_labels[:,i]))

    print('Confusion Matrix:')
    for i in range(6):
        print(confusion_matrix(true_labels[:,i], pred_labels[:,i]))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'training_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_accuracy': np.mean([accuracy_score(true_labels[:,i], pred_labels[:,i]) for i in range(6)]),
            'val_macro_f1': np.mean([f1_score(true_labels[:,i], pred_labels[:,i], average='macro') for i in range(6)]),
            'val_weighted_f1': np.mean([f1_score(true_labels[:,i], pred_labels[:,i], average='weighted') for i in range(6)]),
            'training_time': training_time,
            'val_tim': validation_time
        }
    )

    model_path = 'model_state_dict_'+str(epoch_i)+'.pt'
    torch.save(model.state_dict(), model_path)

print("")
stats_path = 'training_stats_pickle'
pd.DataFrame(training_stats).to_pickle(stats_path)

print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

