import os
import sys
import cv2
import time
from PIL import Image
import imageio
import numpy as np
import torch
from torch.backends import cudnn
import torchvision.transforms as T
from PIL import Image
sys.path.append('.')
from utils.logger import setup_logger
from model import make_model
from config import Config
import cv2
from utils.metrics import cosine_similarity

"""
def visualizer(test_img, camid, top_k = 10, img_size=[128,128]):
    figure = np.asarray(query_img.resize((img_size[1],img_size[0])))
    for k in range(top_k):
        name = str(indices[0][k]).zfill(6)
        img = np.asarray(Image.open(img_path[indices[0][k]]).resize((img_size[1],img_size[0])))
        figure = np.hstack((figure, img))
        title=name
    figure = cv2.cvtColor(figure,cv2.COLOR_BGR2RGB)
    if not os.path.exists(Cfg.LOG_DIR+ "/results/"):
        print('need to create a new folder named results in {}'.format(Cfg.LOG_DIR))
    cv2.imwrite(Cfg.LOG_DIR+ "/results/{}-cam{}.png".format(test_img,camid),figure)
"""


import cv2

cv_net = cv2.dnn.readNetFromTensorflow('/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/JuYeong/person_reid/person-reid-tiny-baseline/JU/pretrained/ssd_inception_v2_coco_2017_11_17/frozen_inference_graph.pb',
                                      '/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/JuYeong/person_reid/person-reid-tiny-baseline/JU/pretrained/ssd_config_01.pbtxt')

labels_to_names = {1:'person',2:'bicycle',3:'car',4:'motorcycle',5:'airplane',6:'bus',7:'train',8:'truck',9:'boat',10:'traffic light',
                    11:'fire hydrant',12:'street sign',13:'stop sign',14:'parking meter',15:'bench',16:'bird',17:'cat',18:'dog',19:'horse',20:'sheep',
                    21:'cow',22:'elephant',23:'bear',24:'zebra',25:'giraffe',26:'hat',27:'backpack',28:'umbrella',29:'shoe',30:'eye glasses',
                    31:'handbag',32:'tie',33:'suitcase',34:'frisbee',35:'skis',36:'snowboard',37:'sports ball',38:'kite',39:'baseball bat',40:'baseball glove',
                    41:'skateboard',42:'surfboard',43:'tennis racket',44:'bottle',45:'plate',46:'wine glass',47:'cup',48:'fork',49:'knife',50:'spoon',
                    51:'bowl',52:'banana',53:'apple',54:'sandwich',55:'orange',56:'broccoli',57:'carrot',58:'hot dog',59:'pizza',60:'donut',
                    61:'cake',62:'chair',63:'couch',64:'potted plant',65:'bed',66:'mirror',67:'dining table',68:'window',69:'desk',70:'toilet',
                    71:'door',72:'tv',73:'laptop',74:'mouse',75:'remote',76:'keyboard',77:'cell phone',78:'microwave',79:'oven',80:'toaster',
                    81:'sink',82:'refrigerator',83:'blender',84:'book',85:'clock',86:'vase',87:'scissors',88:'teddy bear',89:'hair drier',90:'toothbrush',
                    91:'hair brush'}
Cfg = Config()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cudnn.benchmark = True

model = make_model(Cfg, 255)
model.load_param(Cfg.TEST_WEIGHT)

device = 'cuda'
model = model.to(device)
transform = T.Compose([
    T.Resize(Cfg.INPUT_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



log_dir = Cfg.LOG_DIR
logger = setup_logger('{}.test'.format(Cfg.PROJECT_NAME), log_dir)
model.eval()
gallery_feats = torch.load(Cfg.LOG_DIR + '/gfeats.pth')
img_path = np.load('./log/imgpath.npy')
print(gallery_feats.shape, len(img_path))
query_img = Image.open('/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/JuYeong/person_reid/person-reid-tiny-baseline/tools/Hee.jpg')
input = torch.unsqueeze(transform(query_img), 0)
input = input.to(device)
with torch.no_grad():
    query_feat = model(input)
print(query_feat.shape)    


def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat
  
  
def get_detected_img(cv_net, img_array,cnt, query_img,video, score_threshold , is_print=True ):
    query_img =cv2.imread(query_img)
    imgs=[]
    rows = img_array.shape[0]
    cols = img_array.shape[1]
    draw_img = img_array.copy()
    cv_net.setInput(cv2.dnn.blobFromImage(img_array, size=(300, 300), swapRB=True, crop=False))
    start = time.time()
    cv_out = cv_net.forward()
    
    green_color=(0, 255, 0)
    red_color=(0, 0, 255)
    count =0
    # detected 된 object들을 iteration 하면서 정보 추출
    for detection in cv_out[0,0,:,:]:
        score = float(detection[2])
        class_id = int(detection[1])
        # detected된 object들의 score가 0.4 이상만 추출
        if score > score_threshold:
            # detected된 object들은 image 크기가 (300, 300)으로 scale된 기준으로 예측되었으므로 다시 원본 이미지 비율로 계산
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            # labels_to_names 딕셔너리로 class_id값을 클래스명으로 변경. opencv에서는 class_id + 1로 매핑해야함.
            caption = "{}: {:.4f}".format(labels_to_names[class_id], score)
            if class_id ==1:
                #print(str(left)+ ' '+ str(right)+ ' '+ str(top)+' ', str(bottom))
                #print(img_array.shape)
                
                crop_img = img_array[int(top):int(bottom), int(left):int(right)]
                name = ''
                if '준경' in video:
                    name = '2000'
                elif '주영' in video:
                    name = '2001'
                elif '지석' in video:
                    name = '2002'
                elif '성희' in video:
                    name = '2003'            
                imgs.append(crop_img)
                if crop_img.shape[0] >0 and crop_img.shape[1] > 0:
                    #print(crop_img.shape)
                    to_find = Image.fromarray(crop_img)
                    input = torch.unsqueeze(transform(to_find), 0)
                    input = input.to(device)
                    with torch.no_grad():
                        find_feat = model(input)
                    
                    sim = cosine_similarity(query_feat , find_feat)
                    caption = "{}: {:.4f}".format('suspect', float(sim)*60)
                    cv2.imwrite('./11_30_2/'+name+'_'+str(cnt)+'.jpg' , crop_img)
                    count+=1
                    print(sim[0][0])
                    if sim[0][0] >=1.35:
                        cv2.rectangle(draw_img, (int(left), int(top)), (int(right), int(bottom)), color=green_color, thickness=2)
                        cv2.putText(draw_img, caption, (int(left), int(top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, red_color, 2)
                    
    
                print('Detection 수행시간:',round(time.time() - start, 2),"초")
    
    return draw_img

def do_detected_video(cv_net, input_path, output_path,  score_threshold, is_print, query_img):
    
    cap = cv2.VideoCapture(input_path)

    codec = cv2.VideoWriter_fourcc(*'XVID')

    vid_size = (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    vid_fps = cap.get(cv2.CAP_PROP_FPS)

    vid_writer = cv2.VideoWriter(output_path, codec, vid_fps, vid_size) 

    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('총 Frame 갯수:', frame_cnt, )
    cnt =0
    green_color=(0, 255, 0)
    red_color=(0, 0, 255)
    while True:
        hasFrame, img_frame = cap.read()
        if not hasFrame:
            print('더 이상 처리할 frame이 없습니다.')
            break
        
        returned_frame = get_detected_img(cv_net, img_frame, cnt, query_img, input_path , score_threshold=score_threshold, is_print=True)
        cnt+=1
        vid_writer.write(returned_frame)
    # end of while loop

    vid_writer.release()
    cap.release()
    
   #input 동영상, output 동영상, query이미지 



do_detected_video(cv_net, '/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/JuYeong/person_reid/person-reid-tiny-baseline/videos/지석_주영_1.mp4', '/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/JuYeong/person_reid/person-reid-tiny-baseline/JU/data/지석_주영_1_out.mp4', 0.8, False, '/workspace/NIA/NIA_AI_DATASET_2021-ST-GCAE/JuYeong/person_reid/person-reid-tiny-baseline/JU/input/Ji.jpg')

