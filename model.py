import torch  # pytorch
import torch.nn as nn
from torch.autograd import Variable


class LSTMNet(nn.Module):
    def __init__(self, num_classes: int, input_size: int, hidden_size: int, num_layers: int, seq_length: int):
        super(LSTMNet, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)  # lstm
        self.fc_1 = nn.Linear(hidden_size, 128)  # fully connected 1
        self.fc = nn.Linear(128, num_classes)  # fully connected last layer

        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0),
                                   self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0),
                                   self.hidden_size))  # internal state
        # Propagate input through LSTM
        # lstm with input, hidden, and internal state
        output, (hn, cn) = self.lstm(x, (h_0, c_0))

        print(output)

        # reshaping the data for Dense layer next
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.fc_1(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output
        return out


class LSTM():
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length) -> LSTMNet:
        self.model = LSTMNet(num_classes, input_size,
                             hidden_size, num_layers, seq_length)

    def compile(self, learning_rate: float):
        self.criterion = torch.nn.MSELoss()    # mean-squared error for regression
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate)

    def fit(self, num_epochs: int, X: torch.Tensor, y: torch.Tensor,
            batch_size: int, X_validation: torch.Tensor, y_validation: torch.Tensor):

        # Divide into batches
        X_batches = torch.split(X, batch_size)
        y_batches = torch.split(y, batch_size)

        for epoch in range(num_epochs):
            for i, (X, y) in enumerate(zip(X_batches, y_batches)):
                outputs = self.model.forward(X)  # forward pass
                self.optimizer.zero_grad()  # caluclate the gradient, manually setting to 0

                # obtain the loss function
                loss = self.criterion(outputs, y)

                loss.backward()  # calculates the loss of the loss function

                self.optimizer.step()  # improve from loss, i.e backprop

                print(
                    f'Epoch {epoch} of {num_epochs}. Batch {i} of {len(X_batches)}. Loss={loss.item()} \r')

    # def __accuracy(predictions, labels):
    #     classes = torch.argmax(predictions, dim=1)
    #     return torch.mean((classes == labels).float())
