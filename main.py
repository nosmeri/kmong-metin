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
    
    results = model.predict(frame, imgsz=config.IMG_SIZE)
    results[0].show()
    



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

