from torch import nn

class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(in_features=4, out_features=16, bias=True)
        self.fc2 = nn.Linear(in_features=16, out_features=32, bias=True)
        self.fc3 = nn.Linear(in_features=32, out_features=16, bias=True)
        self.fc4 = nn.Linear(in_features=16, out_features=4, bias=True)
        self.af = nn.ReLU()
        self.bn32 = nn.BatchNorm1d(num_features=32)
        self.bn16 = nn.BatchNorm1d(num_features=16)
        self.bn32 = nn.BatchNorm1d(num_features=32)
        self.do   = nn.Dropout1d(p=0.4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 1. Relu
        x = self.fc1(x)
        x = self.af(x)
        x = self.bn16(x)

        x = self.fc2(x)
        x = self.af(x)
        # x = self.bn32(x)

        x = self.fc3(x)
        x = self.af(x)
        # x = self.bn16(x) # on next time
        x = self.do(x)

        x = self.fc4(x)
        x = self.softmax(x)

        # 3. make ins wider
        '''        
        x = self.fc1(x)
        x = self.af(x)
        x = self.bn16(x)

        x = self.fc2(x)
        x = self.af(x)
        x = self.do(x)

        x = self.fc3(x)
        x = self.af(x)
        x = self.do(x)

        x = self.fc4(x)
        x = self.softmax(x)
        '''
        return x