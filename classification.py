from helper import clean_sentences
from helper import make_w_2_i
from helper import make_vocab
from helper import file_read
import torch
from nltk import word_tokenize
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch
import pandas as pd 
import re
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


EMBEDDING_DIM = 200
LEARNING_RATE = 0.1
HIDDEN_DIM = 100
BATCH_SIZE = 50
NUM_EPOCHES = 10


labels, train_sentences = file_read("train.csv")
# print(labels)
train_sentences = train_sentences[:15000]
labels = labels[:15000]
cleaned_train_sentences = clean_sentences(train_sentences)
# print(cleaned_train_sentences[:5])
vocabulary = make_vocab(cleaned_train_sentences)
vocabulary = list(vocabulary)
word_to_index = make_w_2_i(vocabulary)
data_x = []
for sentence in cleaned_train_sentences:
    tokens = word_tokenize(sentence)
    temp_sentence = [word_to_index["<BOS>"]]
    for token in tokens:
        temp_sentence.append(word_to_index[token])
    temp_sentence.append(word_to_index["<EOS>"])
    data_x.append(temp_sentence)
pad_length = int(np.percentile([len(seq) for seq in data_x], 80))
for i in range(len(data_x)):
    if len(data_x[i]) < pad_length:
        data_x[i] += [3] * (pad_length - len(data_x[i]))
    else:
        data_x[i] = data_x[i][:pad_length]


label_set = list(set(labels))
label_to_index = {label: index for index, label in enumerate(label_set)}
# encoded_labels = [[label_to_index[label]]for label in labels]
encoded_labels_train = torch.tensor([label_to_index[label] for label in labels])
# print(data_x)
data_y = encoded_labels_train
data_x = torch.tensor(data_x)
train_dataset = TensorDataset(data_x, data_y)
train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)

labels_test , test_sentences = file_read("test.csv")
test_sentences = test_sentences[:15000]
labels_test = labels_test[:15000]
cleaned_test_sentences = clean_sentences(test_sentences)

# print(word_to_index["fitzpatrick"])
data_x = []
for sentence in cleaned_test_sentences:
    tokens = word_tokenize(sentence)
    temp_sentence = [word_to_index["<BOS>"]]
    for token in tokens:
        if token in vocabulary:
            temp_sentence.append(word_to_index[token])
        else:
            temp_sentence.append(word_to_index["<UNK>"])
    temp_sentence.append(word_to_index["<EOS>"])
    data_x.append(temp_sentence)
pad_length = int(np.percentile([len(seq) for seq in data_x], 80))
for i in range(len(data_x)):
    if len(data_x[i]) < pad_length:
        data_x[i] += [3] * (pad_length - len(data_x[i]))
    else:
        data_x[i] = data_x[i][:pad_length]
        
encoded_labels_test = torch.tensor([label_to_index[label] for label in labels_test])
data_y = encoded_labels_test
data_x = torch.tensor(data_x)


# print(data_y)
# dataset = TensorDataset(data_x, data_y)
# data_loader = DataLoader(dataset, BATCH_SIZE, shuffle=True)


class ELMO(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dimension, hidden_size, batch_size, output_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dimension)
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lstm1 = torch.nn.LSTM(embedding_dimension, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = torch.nn.LSTM(hidden_size * 2, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.scalars = torch.nn.Parameter(torch.ones(3))
        self.linear = torch.nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        embedded = self.embedding(x)

        lstm_out1, _ = self.lstm1(embedded)
        forward_output, backward_output = lstm_out1[:, :, :self.hidden_size], lstm_out1[:, :, self.hidden_size:]
        lstm_out1 = torch.cat((forward_output, backward_output), dim=-1)

        lstm_out2, _ = self.lstm2(lstm_out1)
        forward_output, backward_output = lstm_out2[:, :, :self.hidden_size], lstm_out2[:, :, self.hidden_size:]
        lstm_out2 = torch.cat((forward_output, backward_output), dim=-1)

        elmo_embedding = self.scalars[1] * lstm_out1 + self.scalars[2] * lstm_out2 + self.scalars[0]*embedded
        
        output_logits = self.linear(elmo_embedding)
        return output_logits

# # torch.save(elmo_model.state_dict(), 'bilstm.pt')



# class Classifier(nn.Module):
#     def __init__(self, elmo_model, num_classes):
#         super(Classifier, self).__init__()
#         self.elmo = elmo_model
#         self.fc = nn.Linear(EMBEDDING_DIM * 2, num_classes)  # Assuming EMBEDDING_DIM is the hidden size of ELMo's LSTM

#     def forward(self, x):
#         embedded = self.elmo.embedding(x)

#         lstm_out1, _ = self.elmo.lstm1(embedded)
#         forward_output, backward_output = lstm_out1[:, :, :self.elmo.hidden_size], lstm_out1[:, :, self.elmo.hidden_size:]
#         lstm_out1 = torch.cat((forward_output, backward_output), dim=-1)

#         lstm_out2, _ = self.elmo.lstm2(lstm_out1)
#         forward_output, backward_output = lstm_out2[:, :, :self.elmo.hidden_size], lstm_out2[:, :, self.elmo.hidden_size:]
#         lstm_out2 = torch.cat((forward_output, backward_output), dim=-1)

#         elmo_embedding = self.elmo.scalars[0] * lstm_out1 + self.elmo.scalars[1] * lstm_out2
        
#         output_logits = self.fc(elmo_embedding)
#         return output_logits
    
class LSTMTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,elmo_model):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        
        # self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)
        self.elmo = elmo_model
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden = self.init_hidden()
    

    def forward(self, x):
        self.init_hidden()
        embedded = self.elmo.embedding(x)
        output, _=  self.lstm(embedded)
        # hidden = hidden.squeeze(0)
        # out = self.fc(output[:, -1, :])
        lstm_out_mean = torch.mean(output, dim=1)
        out = self.fc(lstm_out_mean)
        # out = torch.mean(out)

        # print(out)
        out = F.log_softmax(out,dim=1)
        return out
    
    def init_hidden(self):

        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    
num_classes = len(label_set)
elmo_model = ELMO(len(vocabulary), EMBEDDING_DIM, HIDDEN_DIM, BATCH_SIZE, len(vocabulary))
elmo_model.load_state_dict(torch.load('bilstm.pt'))
# classifier = Classifier(elmo_model, num_classes)

model  = LSTMTagger(len(vocabulary),EMBEDDING_DIM,HIDDEN_DIM,num_classes,elmo_model)
optimizer = torch.optim.Adam(elmo_model.parameters(),lr = 0.1)
criterion = nn.CrossEntropyLoss(ignore_index=0)


test_dataset = TensorDataset(data_x, data_y)
test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=True)



print(num_classes)
for epoch in range(10):
    elmo_model.train()
    total_loss = 0.0
    # print(train_loader)
    for input_seq , target_seq in train_loader:

        optimizer.zero_grad()
        output = model(input_seq)
        # print(output.shape)
        # print(target_seq.shape)
        loss = criterion(output.view(-1,output.size(-1)), target_seq.view(-1))
        total_loss += loss.item()


        loss.backward()
        optimizer.step()
    print("DONE")

    print(f"Epoch {epoch+1}/{NUM_EPOCHES}, Loss: {total_loss / len(train_loader)}")


torch.save(elmo_model.state_dict(), 'classification.pt')

# test_accuracy = calculate_accuracy(elmo_model, test_loader)
# print(f"Test Accuracy: {test_accuracy}")
def evaluate(model, dataloader):
    model.eval()  
    correct = 0
    total = 0
    true_labels = []
    predictions = []  
    with torch.no_grad():
        for input_seq, target_seq in dataloader:
            output = model(input_seq)

            _, predicted = torch.max(output, 1)
          
            target = target_seq.view(-1)
            
            correct += (predicted == target_seq).sum().item()
            total += target_seq.size(0)
            true_labels.extend(target_seq.tolist())
            predictions.extend(predicted.tolist())
    # return correct / total
    accuracy = 100 * correct / total
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    confusion = confusion_matrix(true_labels, predictions)
    # print(f'Accuracy: {accuracy}%')
    return accuracy,precision,recall,f1,confusion


accuracy, train_precision, train_recall, train_f1, train_confusion = evaluate(model, train_loader)
print(f"accuracy on Train set: {accuracy}")
print(f"Training Precision: {train_precision}")
print(f"Training Recall: {train_recall}")
print(f"Training F1 Score: {train_f1}")
print(f"Training Confusion Matrix:\n{train_confusion}")

accuracy,test_precision, test_recall, test_f1, test_confusion = evaluate(model, test_loader)
print(f"accuracy on Test set: {accuracy}")
print(f"Testing Precision: {test_precision}")
print(f"Testing Recall: {test_recall}")
print(f"Testing F1 Score: {test_f1}")
print(f"Testing Confusion Matrix:\n{test_confusion}")


# train_accuracy = calculate_accuracy(model, train_loader)
# print(f"Train Accuracy: {train_accuracy*100}")

# test_accuracy = calculate_accuracy(model,test_loader)
# print(f"Test Accuracy: {test_accuracy*100}")

