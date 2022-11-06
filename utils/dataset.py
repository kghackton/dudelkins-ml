from torch.utils.data import Dataset
import torch
import csv

class Defects_Dataset(Dataset):
    def __init__(self, annotation_file):
        self.annotation_file = annotation_file
        with open(self.annotation_file, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter='$')
            self.data = []
            self.label = []
            for i, line in enumerate(reader):
                if i == 0:
                    continue
                #                id_defect                  emergency        id_done_works           id_defense_works
                # print(line)
                # self.data.append([float(line[0])/335.0,   float(line[1]),    float(line[2])/1256.0,   float(line[3])/38.0])
                self.data.append([float(line[0]), float(line[1]), float(line[2]), float(line[3])])
                self.label.append([int(line[4])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.label[idx][0]
        input = self.data[idx]
        return (label, input)
