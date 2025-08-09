from ultralytics import YOLO
import os
from screenshot import get_window_tilte, get_window_img
from config import Config
import pyautogui as pag
from pyautogui import ImageNotFoundException
import pydirectinput as pdi
import keyboard as kb
import time
import sys
import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans
import math

ascii_art_load = r"""


######### ######## ######## ######## ######## ######## ######### ######## ######## ######## ######## 
#  ###  # #      # #      # ##    ## #  ##  # ######## #  ###  # ###  ### ##    ## #     ## ##    ## 
#   #   # #  ##### ###  ### ###  ### #   #  # ######## #   #   # ##    ## #  ##  # #  ##  # #  ##  # 
#       # #    ### ###  ### ###  ### #      # ######## #       # #  ##  # #  ##### #  ##  # #  ##  # 
#  # #  # #  ##### ###  ### ###  ### #      # ######## #  # #  # #      # #  ##### #     ## #  ##  # 
#  ###  # #  ##### ###  ### ###  ### #  #   # ######## #  ###  # #  ##  # #  ##  # #  #  ## #  ##  # 
#  ###  # #      # ###  ### ##    ## #  ##  # ######## #  ###  # #  ##  # ##    ## #  ##  # ##    ## 
######### ######## ######## ######## ######## ######## ######### ######## ######## ######## ######## 

                                              loding...


"""


ascii_art = r"""


######### ######## ######## ######## ######## ######## ######### ######## ######## ######## ######## 
#  ###  # #      # #      # ##    ## #  ##  # ######## #  ###  # ###  ### ##    ## #     ## ##    ## 
#   #   # #  ##### ###  ### ###  ### #   #  # ######## #   #   # ##    ## #  ##  # #  ##  # #  ##  # 
#       # #    ### ###  ### ###  ### #      # ######## #       # #  ##  # #  ##### #  ##  # #  ##  # 
#  # #  # #  ##### ###  ### ###  ### #      # ######## #  # #  # #      # #  ##### #     ## #  ##  # 
#  ###  # #  ##### ###  ### ###  ### #  #   # ######## #  ###  # #  ##  # #  ##  # #  #  ## #  ##  # 
#  ###  # #      # ###  ### ##    ## #  ##  # ######## #  ###  # #  ##  # ##    ## #  ##  # ##    ## 
######### ######## ######## ######## ######## ######## ######### ######## ######## ######## ######## 

                            [PgUP] Select Window [PgDn] Auto Hunt
                            [F3] Save           


"""


os.system("cls")
print("gpu detected.." if torch.cuda.is_available() else "", ascii_art_load)

#--------------------------------------------------------------

config = Config.load()


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = YOLO('./train/weights/best.pt').to(device=device)
dummy = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8)
model.predict(dummy, imgsz=config.IMG_SIZE, conf=0.25, verbose=False)

cls_names = model.names


last_skill_use = {5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
cooldown_sec = {5: 120, 6: 60, 7: 60, 8: 60, 9: 60}


os.system("cls")
print("gpu detected.." if torch.cuda.is_available() else "", ascii_art)

pdi.PAUSE = 0

#--------------------------------------------------------------

def is_skill_ready(key):
    return (time.time() - last_skill_use[key]) >= cooldown_sec[key]

def use_skill(key):
    press_key(str(key))
    last_skill_use[key] = time.time()
    print(f"[스킬] {key}번 사용")

def get_box(boxes, region):
    xyxy = boxes.xyxy.cpu().numpy()
    cls  = boxes.cls.cpu().numpy()
    conf = boxes.conf.cpu().numpy()

    screen_boxes = []
    x_off, y_off = region[0], region[1]

    for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
        sx1, sy1, sx2, sy2 = x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off
        cx, cy = (sx1 + sx2) / 2, (sy1 + sy2) / 2
        if int(c) == 0:
            continue # 시체 제외
        screen_boxes.append({
            "xyxy_screen": (int(sx1), int(sy1), int(sx2), int(sy2)),
            "center_screen": (int(cx), int(cy)),
            "cls": int(c)-1,
            "conf": float(p),
        })

    # cls==2만 따로 필터링
    cls2_boxes = [b for b in screen_boxes if b["cls"] == 2]
    if cls2_boxes:
        # 확률 가장 높은 cls2 하나만
        best_cls2 = max(cls2_boxes, key=lambda b: b["conf"])
        # cls==2가 아닌 박스들과 합쳐서 반환
        others = [b for b in screen_boxes if b["cls"] != 2]
        return others + [best_cls2]
    else:
        # cls2 없으면 전체 그대로
        return screen_boxes



def split_cluster_and_singles_sklearn(
    screen_boxes,
    target_cls=1,
    min_cluster_size=3,
    max_k=3,
):
    """
    cls==target_cls 만 골라서:
      - 멤버 수 >= min_cluster_size 군집 → '군집 타깃' (가중치합 내림차순)
      - 그 외(멤버 수 < min_cluster_size) → '개별 타깃' (conf 내림차순)
    return: (cluster_targets, single_targets)
      cluster_targets: [(x, y), ...]  # 군집 중심(가중치 평균)
      single_targets : [(x, y), ...]  # 개별 중심
    """
    # 1) 대상 필터링
    targets = [b for b in screen_boxes if b["cls"] == target_cls]
    if not targets:
        return [], []

    pts  = np.array([b["center_screen"] for b in targets], dtype=np.float32)  # (N,2)
    conf = np.array([b["conf"]          for b in targets], dtype=np.float32)  # (N,)
    N = len(pts)

    # 2) 너무 적으면 전부 개별 처리
    if N < min_cluster_size:
        singles_order = np.argsort(-conf)  # conf 내림차순
        single_targets = [tuple(map(int, pts[i])) for i in singles_order]
        return [], single_targets

    # 3) 적당한 k 선정 (대략 'min_cluster_size 당 1군집' 목표 + 상한)
    est_k = int(np.ceil(N / float(min_cluster_size)))
    k = max(2, min(max_k, est_k, N))  # [2, max_k] 범위, N 초과 금지

    # 4) KMeans (conf를 가중치로 반영)
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=0)
    kmeans.fit(pts, sample_weight=conf)
    labels  = kmeans.labels_
    centers = kmeans.cluster_centers_  # (k,2)

    # 5) 군집/개별 분리
    cluster_targets = []
    single_indices  = []

    for i in range(k):
        idxs = np.where(labels == i)[0]
        if len(idxs) >= min_cluster_size:
            wsum = float(conf[idxs].sum())
            # 가중치 중심(centroid 보정)
            cx, cy = (pts[idxs] * conf[idxs][:, None]).sum(axis=0) / max(wsum, 1e-6)
            cluster_targets.append((wsum, int(cx), int(cy)))
        else:
            single_indices.extend(idxs.tolist())

    # 6) 정렬
    cluster_targets.sort(key=lambda t: t[0], reverse=True)        # 가중치합 내림차순
    single_indices.sort(key=lambda i: conf[i], reverse=True)      # conf 내림차순

    cluster_targets = [(x, y) for _, x, y in cluster_targets]
    single_targets  = [tuple(map(int, pts[i])) for i in single_indices]

    return cluster_targets, single_targets


def nearby_units(center, screen_boxes, target_classes=(0, 1), radius=150):
    cx, cy = center
    res=[]
    for b in screen_boxes:
        if b["cls"] in target_classes:
            bx, by = b["center_screen"]
            dist = math.hypot(bx - cx, by - cy)
            if dist <= radius:
                res.append(b)
    return res

def press_key(key):
    pdi.keyDown(str(key))
    time.sleep(0.1)
    pdi.keyUp(str(key))
    time.sleep(0.1)


def auto_hunt():
    if not config.WINDOW_TITLE:
        print("창을 선택해주세요")
        time.sleep(2)
        return
    
    no_cnt=0
    while True:
        if kb.is_pressed("pagedown"):
            return
        pdi.moveTo(100, 100)  # 마우스 위치 초기화
        time.sleep(0.05)

        img, region=get_window_img(config.WINDOW_TITLE)
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        try:
            pag.locate("sinsung.png", img, confidence=0.8)
        except ImageNotFoundException:
            print("신성 사용")
            press_key("3")

        
        start_time = time.perf_counter()
        results = model.predict(frame, imgsz=config.IMG_SIZE, save=False, show=False, conf=config.CONFIDENCE)
        print("추론시간:", (time.perf_counter() - start_time)*1000, "ms")
        result = results[0]
        

        #result.show()

        screen_boxes = get_box(result.boxes, region)

        cls0 = [b for b in screen_boxes if b["cls"] == 0] # monster
        cls1 = [b for b in screen_boxes if b["cls"] == 1] # ogre
        cls2 = [b for b in screen_boxes if b["cls"] == 2] # player
        cls2 = cls2[0] if cls2 else None

        if not cls0 and not cls1:
            no_cnt += 1
            if no_cnt >= 5:
                print("몬스터가 없습니다. 텔레포트합니다.")
                no_cnt = 0
                press_key("0")
                time.sleep(2)
                continue
            else:
                print("몬스터가 없습니다. 0.2초 후 다시 시도합니다.")
                time.sleep(0.2)
                continue
        

        skill_used=False
        cx, cy = (region[0] + region[2]) // 2, (region[1] + region[3]) // 2
        units = nearby_units((cx, cy), screen_boxes, target_classes=(0, 1), radius=config.PLAYER_RADIUS)
        if len(units) >= 2:
            print("근처에 3마리 이상 몬스터가 있습니다.")
            pdi.moveTo(cx, cy, duration=0.1)
            skill_used=True
            if is_skill_ready(5):
                use_skill(5)
            elif is_skill_ready(6):
                use_skill(6)
            elif is_skill_ready(9):
                use_skill(9)
            else:
                print("쿨타임 중입니다.")
                skill_used=False

        for box in cls0:
            if skill_used and box in units:
                continue  # 이미 스킬 사용한 몬스터는 건너뜀

            cx, cy = box["center_screen"]

            pdi.moveTo(cx, cy, duration=0.1)
            print("몬스터 발견:", cls_names[box["cls"]], "확률:", box["conf"])
            press_key("4")  # 공격 키

        clusters, singles = split_cluster_and_singles_sklearn(
            screen_boxes,
            target_cls=1,        # <- cls==1만 대상
            min_cluster_size=3,  # <- 3마리 이상만 군집으로 인정
            max_k=3,             # <- 군집 최대 3개까지 찾음
        )


        for cx, cy in clusters:
            pdi.moveTo(cx, cy, duration=0.1)
            print("군집 발견:", cx, cy)
            if is_skill_ready(7):
                use_skill(7)
            elif is_skill_ready(8):
                use_skill(8)
            else:
                press_key("4")
                print("공격: 4")

        
        for cx, cy in singles:
            pdi.moveTo(cx, cy, duration=0.1)
            print("개별 발견:", cx, cy)
            press_key("4")

        time.sleep(0.1)
    
    



while True:
    if kb.is_pressed("pageup"):
        config.WINDOW_TITLE=get_window_tilte()
        print()
        print("선택됨:", config.WINDOW_TITLE)
        time.sleep(1)

        os.system("cls")
        print(ascii_art)
    elif kb.is_pressed("pagedown"):
        print("3초 후 자동사냥이 시작됩니다")
        time.sleep(3)

        auto_hunt()

        os.system("cls")
        print(ascii_art)

        time.sleep(2)
    elif kb.is_pressed("F3"):
        config.save()
        print("저장 완료..")
        time.sleep(1)

        os.system("cls")
        print(ascii_art)

