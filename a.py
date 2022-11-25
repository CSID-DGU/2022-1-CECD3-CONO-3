import os
file_path = '/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/JuYeong/person_reid/person-reid-tiny-baseline/JU/output'
file_names = os.listdir(file_path)
print(file_names[0:20])

i='_c1s1_000451_01.jpg'
for name in file_names:
    src=os.path.join(file_path, name)
    name = name.split('.')[0]
    dst=name+i
    dst=os.path.join(file_path,dst)
    os.rename(src,dst)
    