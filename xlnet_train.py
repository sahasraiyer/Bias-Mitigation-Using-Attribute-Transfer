#This is the code for finetuning a XLNet classifier so that it can act as a pretrained model for the Reverse Attention model

import pandas as pd
from transformers import XLNetTokenizer, XLNetForSequenceClassification,AdamW
import numpy as np
import torch
from tqdm import tqdm

assert torch.cuda.is_available()

# Get the GPU device name.
device_name = torch.cuda.get_device_name()
n_gpu = torch.cuda.device_count()
print(f"Found device: {device_name}, n_gpu: {n_gpu}")
device = torch.device("cuda")

model_version = 'xlnet-base-cased'
# model_version = 'bert-base-cased'

tokenizer = XLNetTokenizer.from_pretrained(model_version,padding_side ="right")


# Create the train and valid dataset
train_sent = pd.read_csv("sent_train.csv")
train_class = pd.read_csv("class_train.csv")

valid_sent = pd.read_csv("sent_train.csv")
valid_class = pd.read_csv("class_train.csv")

a = np.array(train_sent['0'])
train_inputs = tokenizer.batch_encode_plus(a, return_tensors='pt', add_special_tokens=True, padding =True, truncation = True,max_length = 512)
train_input_ids = train_inputs['input_ids']
train_attention_masks = train_inputs['attention_mask']
train_labels = torch.tensor(np.array(train_class['0']),dtype = int)

train_set = [(train_input_ids[i], train_attention_masks[i], train_labels[i]) for i in range(len(train_input_ids))]

    
b = np.array(train_sent['0'])
valid_inputs = tokenizer.batch_encode_plus(b, return_tensors='pt', add_special_tokens=True, padding =True, truncation = True,max_length = 512)
valid_input_ids = valid_inputs['input_ids']
valid_attention_masks = valid_inputs['attention_mask']
valid_labels = torch.tensor(np.array(valid_class['0']),dtype = int)
valid_set = [(valid_input_ids[i], valid_attention_masks[i], valid_labels[i]) for i in range(len(valid_input_ids))]


print("Done with tokenization and data loading")

model = XLNetForSequenceClassification.from_pretrained(model_version,num_labels = 2, output_attentions=False,output_hidden_states = False)
model.to(device)

# Hyper parameters
batch_size = 2
epochs = 1
optimizer = AdamW(model.parameters(),
                  lr = 5e-5, # args.learning_rate - default is 5e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8
                )



def get_validation_performance(val_set):
    # Put the model in evaluation mode
    model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0

    num_batches = int(len(val_set)/batch_size) + 1

    total_correct = 0

    for i in range(num_batches):

      end_index = min(batch_size * (i+1), len(val_set))

      batch = val_set[i*batch_size:end_index]
      
      if len(batch) == 0: continue

      input_id_tensors = torch.stack([data[0] for data in batch])
      input_mask_tensors = torch.stack([data[1] for data in batch])
      label_tensors = torch.stack([data[2] for data in batch])
      
      # Move tensors to the GPU
      b_input_ids = input_id_tensors.to(device)
      b_input_mask = input_mask_tensors.to(device)
      b_labels = label_tensors.to(device)
        
      # Tell pytorch not to bother with constructing the compute graph during
      # the forward pass, since this is only needed for backprop (training).
      with torch.no_grad():        

        # Forward pass, calculate logit predictions.
        outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask,
                                labels=b_labels)
        loss = outputs.loss
        logits = outputs.logits
            
        # Accumulate the validation loss.
        total_eval_loss += loss.item()
        
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the number of correctly labeled examples in batch
        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        num_correct = np.sum(pred_flat == labels_flat)
        total_correct += num_correct
        
    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_correct / len(val_set)
    return avg_val_accuracy

import random

# training loop

# For each epoch...
for epoch_i in range(0, epochs):
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode.
    model.train()

    # For each batch of training data...
    num_batches = int(len(train_set)/batch_size) + 1

    for i in tqdm(range(num_batches)):
      end_index = min(batch_size * (i+1), len(train_set))

      batch = train_set[i*batch_size:end_index]
      if len(batch) == 0: continue

      input_id_tensors = torch.stack([data[0] for data in batch])
      input_mask_tensors = torch.stack([data[1] for data in batch])
      label_tensors = torch.stack([data[2] for data in batch])

      # Move tensors to the GPU
      b_input_ids = input_id_tensors.to(device)
      b_input_mask = input_mask_tensors.to(device)
      b_labels = label_tensors.to(device)

      # Clear the previously calculated gradient
      model.zero_grad()        

      # Perform a forward pass (evaluate the model on this training batch).
    #   print(b_input_ids.shape)
    #   print(b_input_mask.shape)
    #   print(b_labels.shape)
      outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels,return_dict = True)
      loss = outputs.loss
      logits = outputs.logits

      total_train_loss += loss.item()

      # Perform a backward pass to calculate the gradients.
      loss.backward()

      # Update parameters and take a step using the computed gradient.
      optimizer.step()

      if i%100==0:
          torch.save({
            'epoch': epoch_i,
            'iter':i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}, f'saved_models/epoch_{epoch_i}_iter_{i}')
        
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set. Implement this function in the cell above.
    print(f"Total loss: {total_train_loss}")
    val_acc = get_validation_performance(valid_set)
    print(f"Validation accuracy: {val_acc}")
    
print("")
print("Training complete!")
