from utils.model import ClassificationModel as net
from torch.utils.data.dataset import random_split
from utils.dataset import Defects_Dataset
from torch.utils.data import DataLoader
import random
import torch
import time


def train(dataloader):
    model.train()
    total_acc, total_count, total_loss = 0, 0, 0
    log_interval = 200
    start_time = time.time()

    for idx, (label, input) in enumerate(dataloader):
        # print(idx)
        optimizer.zero_grad()
        predicted_label = model(input)
        # print(predicted_label.shape, label.shape)
        loss = criterion(predicted_label, label)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        # print(predicted_label, predicted_label.argmax(1), label)
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_loss += loss
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:

            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}| loss {}'.format(epoch, idx, len(dataloader), total_acc/total_count, total_loss/total_count))
            total_acc, total_count, total_loss = 0, 0, 0
            start_time = time.time()


def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, input) in enumerate(dataloader):
            predicted_label = model(input)
            loss = criterion(predicted_label, label)
            # print(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count


def collate_batch_train(batch):
    label_list, input_list = [], []
    for (_label, _input) in batch:
        label_list.append(_label)
        __input = _input.copy()
        # __input[0] = (__input[0] + (random.random() - 0.5)) / 335.0
        # __input[2] = (__input[2] + (random.random() - 0.5)) / 1256.0
        # __input[3] = (__input[3] + (random.random() - 0.5)) / 38.0
        __input[0] = __input[0] / 335.0
        __input[2] = __input[2] / 1256.0
        __input[3] = __input[3] / 38.0
        input_list.append(__input)
    label_list = torch.tensor(label_list, dtype=torch.long)
    input_list = torch.tensor(input_list, dtype=torch.float)
    # print(f'label.shape: {label_list.shape}\tinput.shape: {input_list.shape}')
    return label_list.to(device), input_list.to(device)

def collate_batch_val(batch):
    label_list, input_list = [], []
    for (_label, _input) in batch:
        label_list.append(_label)
        __input = _input.copy()
        __input[0] = __input[0] / 335.0
        __input[2] = __input[2] / 1256.0
        __input[3] = __input[3] / 38.0
        input_list.append(__input)
    label_list = torch.tensor(label_list, dtype=torch.long)
    input_list = torch.tensor(input_list, dtype=torch.float)
    # print(f'label.shape: {label_list.shape}\tinput.shape: {input_list.shape}')
    return label_list.to(device), input_list.to(device)

PATH_CSV = '../dataset.csv'
EPOCHS = 200
LR = 0.001
BATCH_SIZE = 512

# choose device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create dataset and dataloader and tokenizer with vocab
print('Load Data...')
dataset = Defects_Dataset(PATH_CSV)
num_train = int(len(dataset) * 0.8)
split_train, split_valid = random_split(dataset, [num_train, len(dataset) - num_train])
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
train_dataloader = DataLoader(split_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch_train)
valid_dataloader = DataLoader(split_valid, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch_val)

# load model
print('Load Model...')
num_class = 4
model = net().float().to(device)
print(model)

# define train options
criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=LR)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
# total_accu = None
total_accu = -500

# train and validation
for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    # if total_accu is not None and total_accu > accu_val:
    #   scheduler.step()
    # if total_accu is not None and epoch % 3 == 0:
    scheduler.step()
    # else:
    #    total_accu = accu_val
    #    torch.save(model, './best.pt')
    if accu_val >= total_accu:
        print('**********Saving Model**********')
        total_accu = accu_val
        torch.save(model, './best.pt')
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))
    print('-' * 59)



# for label, text in dataset:
#     print(tokenizer(text).text)
#     print([token.text for token in tokenizer(text)])