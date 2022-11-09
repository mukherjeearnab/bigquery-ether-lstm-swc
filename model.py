import torch  # pytorch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
print(device)


class LSTMNet(nn.Module):
    def __init__(self, num_classes: int, input_size: int, hidden_size: int, num_layers: int, seq_length: int):
        super(LSTMNet, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = 1  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length

        self.embeddings = nn.Embedding(
            150, 50, padding_idx=0)
        self.lstm = nn.LSTM(input_size=50, hidden_size=hidden_size,
                            num_layers=1, batch_first=True)  # lstm
        self.fc_1 = nn.Linear(hidden_size, 128)  # fully connected 1
        self.fc = nn.Linear(128, num_classes)  # fully connected last layer

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):

        x = self.embeddings(x.int())

        h_0 = Variable(torch.zeros(self.num_layers, x.size(0),
                                   self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0),
                                   self.hidden_size))  # internal state
        # Propagate input through LSTM
        # lstm with input, hidden, and internal state
        output, (hn, cn) = self.lstm(x, (h_0, c_0))

        # print(output)

        # reshaping the data for Dense layer next
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.fc_1(out)  # first Dense

        # print("DenseInt", out)

        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output
        out = self.sigmoid(out)
        return out


class LSTM():
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length) -> LSTMNet:
        self.model = LSTMNet(num_classes, input_size,
                             hidden_size, num_layers, seq_length)

        self.model.to(device)

    def compile(self, learning_rate: float):
        self.criterion = torch.nn.BCELoss()    # mean-squared error for regression
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate)

    def fit(self, num_epochs: int, X: torch.Tensor, y: torch.Tensor,
            batch_size: int, X_validation: torch.Tensor, y_validation: torch.Tensor):

        X.to(device)
        y.to(device)

        # # Divide into batches
        # X_batches = torch.split(X, batch_size)
        # y_batches = torch.split(y, batch_size)

        for epoch in range(num_epochs):
            accuracy = 0.0
            for (sequence, label) in zip(X, y):
                outputs = self.model.forward(sequence)  # forward pass
                self.optimizer.zero_grad()  # caluclate the gradient, manually setting to 0

                # print(outputs)

                # outputs = (outputs > 0.5)

                # obtain the loss function
                loss = self.criterion(outputs[0], label)

                loss.backward()  # calculates the loss of the loss function

                self.optimizer.step()  # improve from loss, i.e backprop

                # print(outputs, y)

                output = (outputs > 0.5).float()
                accuracy += (output == label)

            print(X.shape)

            accuracy = accuracy/y.shape[0]  # self.__evaluate(X=X, y=y)

            print(
                f'Epoch {epoch} of {num_epochs}. Loss={loss.item()} Acc={accuracy}', end='\n')

    # def __accuracy(predictions, labels):
    #     classes = torch.argmax(predictions, dim=1)
    #     return torch.mean((classes == labels).float())

    def __evaluate(self, X: torch.Tensor, y: torch.Tensor, use_cuda=False):
        self.model.eval()
        with torch.no_grad():
            acc = .0
            if use_cuda:
                X = X.cuda()
                y = y.cuda()
            y_preds = self.model(X)
            y_preds = (y_preds > 0.5).float()
            acc = (y_preds == y).sum()/float(y_preds.shape[0])
        return acc.detach().item()
