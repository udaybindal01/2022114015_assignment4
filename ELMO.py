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
import numpy as np

EMBEDDING_DIMENSIONS = 200
LEARNING_RATE = 0.1
HIDDEN_DIMENSIONS = 100
BATCH_SIZE = 50
NUM_EPOCHS = 10

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ###################################################################
    ##### Preparing the train dataset for generating word vectors #####
    ###################################################################

    _, train_sentences = file_read("train.csv")
    train_sentences = train_sentences[:15000]
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
    
    data_y = torch.tensor(data_x)
    data_x = torch.tensor(data_x)
    # print(data_x.shape)
    # print(len(vocabulary))
    dataset = TensorDataset(data_x, data_y)
    data_loader = DataLoader(dataset, BATCH_SIZE, shuffle=True)

    class CombinationFunction(nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()
            # Define the layers of the combination function
            self.fc1 = nn.Linear(input_size, output_size)
            self.activation = nn.ReLU()
            self.fc2 = nn.Linear(output_size, output_size)
            self.scalars = torch.nn.Parameter(torch.ones(3))

        def forward(self, e0, e1, e2):

            # combined = torch.cat((e0, e1, e2), dim=1)
            combined = self.scalars[1] * e1 + self.scalars[2] * e2 + self.scalars[0] * e0

            out = self.fc1(combined)
            out = self.activation(out)
            out = self.fc2(out)
            return out
        
    class ELMO(torch.nn.Module):
        def __init__(self, vocab_size, embedding_dimension, hidden_size, batch_size, output_size):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, embedding_dimension)
            self.hidden_size = hidden_size
            self.batch_size = batch_size
            self.lstm1 = torch.nn.LSTM(embedding_dimension, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
            self.lstm2 = torch.nn.LSTM(hidden_size * 2, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
            # self.scalars = torch.nn.Parameter(torch.ones(2))
            self.scalars = torch.nn.Parameter(torch.ones(3))
            # self.scalars = torch.nn.Parameter(torch.tensor([0.1, -0.5, -0.03]))
            self.linear = torch.nn.Linear(hidden_size * 2, output_size)
            # self.combination_function = CombinationFunction(hidden_size*2, output_size)
            

        def forward(self, x):

            embedded = self.embedding(x)

            lstm_out1, _ = self.lstm1(embedded)

            forward_output, backward_output = lstm_out1[:, :, :self.hidden_size], lstm_out1[:, :, self.hidden_size:]
            lstm_out1 = torch.cat((forward_output, backward_output), dim=-1)

            lstm_out2, _ = self.lstm2(lstm_out1)

            forward_output, backward_output = lstm_out2[:, :, :self.hidden_size], lstm_out2[:, :, self.hidden_size:]
            lstm_out2 = torch.cat((forward_output, backward_output), dim=-1)

            elmo_embedding =  self.scalars[1] * lstm_out1 + self.scalars[2] * lstm_out2 + self.scalars[0] * embedded

            output_logits = self.linear(elmo_embedding)
            # output_logits = self.combination_function(lstm_out1, lstm_out2, embedded)

            return output_logits
    

    elmo_model = ELMO (
        vocab_size=len(vocabulary),
        embedding_dimension=EMBEDDING_DIMENSIONS,
        hidden_size=HIDDEN_DIMENSIONS,
        batch_size=BATCH_SIZE,
        output_size=len(vocabulary)
    ).to(device)
    print(len(vocabulary))
 
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(elmo_model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        elmo_model.train()
        epoch_loss = 0.0
        no_batches = 0
        for inputs, targets in data_loader:
            
            optimizer.zero_grad()
            inputs = torch.LongTensor(inputs)
            inputs = inputs.to(device)
            outputs = elmo_model(inputs)
        
            outputs = outputs.view(-1, outputs.size(-1)) # reshape to [batch_size * seq_len, num_classes]
            targets = targets.view(-1) # reshape to [batch_size * seq_len]
           
            targets = targets.to(device)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            no_batches += 1
            if no_batches % 400 == 0:
                print(f"{no_batches}/{len(data_x)/BATCH_SIZE}, loss: {loss.item()}")
        epoch_loss = epoch_loss/no_batches
        print(f"For Epoch: {epoch + 1}/{NUM_EPOCHS}, loss: {epoch_loss}")
    
    torch.save(elmo_model.state_dict(), 'bilstm.pt')
    # elmo_word_vectors = elmo_model.embedding.weight.data
    # torch.save(elmo_word_vectors, 'bilstm.pt')
    best_lambdas = elmo_model.scalars.detach().numpy()
    print("Best λs:", best_lambdas)

if __name__ == '__main__':
    main()
