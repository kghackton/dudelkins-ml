import json
import os

with open('./id_defect_enumerate.json', 'r') as jsonfile1:
    id_defect_dict = json.load(jsonfile1)

with open('./id_done_works_enumerate.json', 'r') as jsonfile2:
    id_done_works_dict = json.load(jsonfile2)

with open('./id_done_defense_works_enumerate.json', 'r') as jsonfile3:
    id_done_defense_works = json.load(jsonfile3)

_id_defect_dict = {}
_id_done_works_dict = {}
_id_done_defense_works_dict = {}

for k1, v1 in id_defect_dict.items():
    for k2, v2 in v1.items():
        _id_defect_dict[k2] = k1
    # print(key1, value1, key2, value2)

for k1, v1 in id_done_works_dict.items():
    for k2, v2 in v1.items():
        _id_done_works_dict[k2] = k1

for k1, v1 in id_done_defense_works.items():
    for k2, v2 in v1.items():
        _id_done_defense_works_dict[k2] = k1

with open('./_id_defect_dict.json', 'w', encoding='utf-8') as jsonfile:
    json.dump(_id_defect_dict, jsonfile, sort_keys=True, ensure_ascii=False)

with open('./_id_done_works_dict.json', 'w', encoding='utf-8') as jsonfile:
    json.dump(_id_done_works_dict, jsonfile, sort_keys=True, ensure_ascii=False)

with open('./_id_done_defense_works_dict.json', 'w', encoding='utf-8') as jsonfile:
    json.dump(_id_done_defense_works_dict, jsonfile, sort_keys=True, ensure_ascii=False)