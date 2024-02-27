#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install transformers')
get_ipython().system('pip install pytorch')
get_ipython().system('pip install colorama')
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#print(tokenizer)
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

sentence_A = "I am from kolkata "

sentence_B ="cloud computing" 
tokenized = tokenizer(sentence_A, sentence_B, return_tensors='pt')
tokenized.keys()
tokenized
labels = torch.LongTensor([0])
labels
predict = model(**tokenized, labels=labels)
predict.loss
predict = model(**tokenized, labels=labels)
predict.keys()
print(predict)
prediction = torch.argmax(predict.logits)
if prediction == 0:
  print('\033[2;31;43mTruepair\033[0;0m')
else:
  print('\033[2;31;43mFalsepair\033[0;0m')


# In[ ]:




