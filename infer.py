import torch
import json
import time

def predict(message):
    with torch.no_grad():

        id_defect = float(id_defect_dict[message['id_defect']])
        id_emergency = 0.0 if message['id_emergency'] == 'normal' else 1.0
        done_works = [float(id_done_works_dict[x]) for x in message['id_done_works'].split(',')]
        if message['id_security_works'] == '0':
            security_works = [0]
        else:
            security_works = [float(id_done_security_works[x]) for x in message['id_security_works'].split(',')]
        label = label_map[message['result']]
        # print(id_defect, id_emergency, done_works, security_works, label)

        batch_input = []
        batch_label = []
        for dw in done_works:
            for ddw in security_works:
                sample = torch.tensor((id_defect/335.0, id_emergency, dw / 1256.0, ddw / 38.0), dtype=torch.float)
                batch_input.append(sample)
                labels = torch.tensor(label, dtype=torch.long)
                batch_label.append(labels)

                # print(id_defect, id_emergency, dw, ddw, label)
        batch_input = torch.stack(batch_input, dim=0)
        batch_label = torch.stack(batch_label, dim=0)
        # print(batch_input)
        # print(batch_label)

        tt0 = time.time()
        output = model(batch_input)
        tt1 = time.time()
        print(tt1-tt0)
        # print(output)

        acc = (output.argmax(1) == batch_label).sum().item()
        if acc > 0:
            return True
        return False

t0 = time.time()

label_map = {
    'resolved': 0,
    'consulted': 1,
    'reject': 2,
    'cancel': 3
}

# define model
model = torch.load('./best_878.pt')
model.to('cpu').eval()

with open('./_id_defect_dict.json', 'r') as jsonfile1:
    id_defect_dict = json.load(jsonfile1)

with open('./_id_done_works_dict.json', 'r') as jsonfile2:
    id_done_works_dict = json.load(jsonfile2)

with open('./_id_done_security_works_dict.json', 'r') as jsonfile3:
    id_done_security_works = json.load(jsonfile3)

# '0'
# '15378,18346,18426'
message = {'id_defect': '2229', 'id_emergency': 'emergency', 'id_done_works': '15420', 'id_security_works': '15378,18346,18426', 'result': 'consulted'}

pred = predict(message)
# print(pred)
t1 = time.time()

print(t1-t0)


