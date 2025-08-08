from ultralytics import YOLO
import os
from screenshot import get_window_tilte, get_window_img
from config import Config
import pyautogui as pag
from pyautogui import ImageNotFoundException
import keyboard as kb
import time
import sys
import cv2
import numpy as np
import torch

os.system("cls")

ascii_art = r"""


######### ######## ######## ######## ######## ######## ######### ######## ######## ######## ######## 
#  ###  # #      # #      # ##    ## #  ##  # ######## #  ###  # ###  ### ##    ## #     ## ##    ## 
#   #   # #  ##### ###  ### ###  ### #   #  # ######## #   #   # ##    ## #  ##  # #  ##  # #  ##  # 
#       # #    ### ###  ### ###  ### #      # ######## #       # #  ##  # #  ##### #  ##  # #  ##  # 
#  # #  # #  ##### ###  ### ###  ### #      # ######## #  # #  # #      # #  ##### #     ## #  ##  # 
#  ###  # #  ##### ###  ### ###  ### #  #   # ######## #  ###  # #  ##  # #  ##  # #  #  ## #  ##  # 
#  ###  # #      # ###  ### ##    ## #  ##  # ######## #  ###  # #  ##  # ##    ## #  ##  # ##    ## 
######### ######## ######## ######## ######## ######## ######### ######## ######## ######## ######## 

                            [F1] Select Window   [F2] Auto Hunt
                            [F3] Save            [F4] Exit


"""

print("gpu detected.." if torch.cuda.is_available() else "", ascii_art)

#--------------------------------------------------------------


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = YOLO('./train/weights/best.pt').to(device=device)
config = Config.load()

#--------------------------------------------------------------

def get_pos(region, origin_pos):
    return region[0]+origin_pos[0],region[1]+origin_pos[1]


def get_box(boxes, region):
    xyxy = boxes.xyxy.cpu().numpy()      # (N, 4) in [x1,y1,x2,y2] on the cropped frame
    cls  = boxes.cls.cpu().numpy()
    conf = boxes.conf.cpu().numpy()

    screen_boxes = []
    x_off, y_off = region[0], region[1]  # window top-left offset on the screen

    for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
        # 화면 전체 기준 좌표
        sx1, sy1, sx2, sy2 = x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off
        # 화면에서의 중심점(필요 시 클릭/조준 등)
        cx, cy = (sx1 + sx2) / 2, (sy1 + sy2) / 2

        screen_boxes.append({
            "xyxy_screen": (int(sx1), int(sy1), int(sx2), int(sy2)),
            "center_screen": (int(cx), int(cy)),
            "cls": int(c),
            "conf": float(p),
        })
    return screen_boxes




def auto_hunt():
    if not config.WINDOW_TITLE:
        print("창을 선택해주세요")
        time.sleep(2)
        return
    
    img, region=get_window_img(config.WINDOW_TITLE)
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    try:
        pag.locate("sinsung.png", img, confidence=0.8)
    except ImageNotFoundException:
        print("신성 사용")
        pag.press("3")
    results = model.predict(frame, imgsz=config.IMG_SIZE, save=False, show=False)
    screen_boxes = get_box(results[0].boxes, region)

    for box in screen_boxes:
        x1, y1, x2, y2 = box["xyxy_screen"]
        cx, cy = box["center_screen"]
        cls = box["cls"]

        if cls == 0:
            pag.moveTo(cx, cy, duration=0.05)
            print("몬스터 발견:", box["cls"], "확률:", box["conf"])
            pag.press("4")
            print("공격: 4")
            time.sleep(0.1)
    
    



while True:
    if kb.is_pressed("F1"):
        config.WINDOW_TITLE=get_window_tilte()
        print()
        print("선택됨:", config.WINDOW_TITLE)
        time.sleep(1)

        os.system("cls")
        print(ascii_art)
    elif kb.is_pressed("F2"):
        print("3초 후 자동사냥이 시작됩니다")
        time.sleep(3)

        auto_hunt()

        #os.system("cls")
        #print(ascii_art)
    elif kb.is_pressed("F3"):
        config.save()
        print("저장 완료..")
        time.sleep(1)

        os.system("cls")
        print(ascii_art)
    elif kb.is_pressed("F4"):
        sys.exit(1)

