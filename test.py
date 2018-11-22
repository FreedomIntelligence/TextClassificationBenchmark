# -*- coding: utf-8 -*-

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

bert_dir = "d:/dataset/bert/uncased_L-12_H-768_A-12/vocab.txt"

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained(bert_dir)

# Tokenized input
text = "Benyou's research interests include (but not limited) information retrieval, natural language processing, machine learning and quantum mechanics"
tokenized_text = tokenizer.tokenize(text)

# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = 18
tokenized_text[masked_index] = '[MASK]'
#assert tokenized_text == ['who', 'was', 'jim', 'henson', '?', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer']

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = len(indexed_tokens)*[0]

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# Predict hidden states features for each layer
encoded_layers, _ = model(tokens_tensor)
# We have a hidden states for each of the 12 layers in model bert-base-uncased
assert len(encoded_layers) == 12


# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()


# Tokenized input
text = "I love china  and taiwan"
tokenized_text = tokenizer.tokenize(text)

# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = 3
tokenized_text[masked_index] = '[MASK]'
#assert tokenized_text == ['who', 'was', 'jim', 'henson', '?', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer']

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = len(indexed_tokens)*[0]

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
# Predict all tokens
predictions = model(tokens_tensor, segments_tensors)

# confirm we were able to predict 'henson'
#predicted_index = torch.argmax(predictions[0, masked_index]).item()
#predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
#assert predicted_token[0] == 'henson'

predicted_indexes = torch.topk(predictions[0, masked_index],5)[1].data.numpy()
predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_indexes)
print(predicted_tokens)