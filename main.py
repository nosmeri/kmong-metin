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

# =============================
# ASCII ART
# =============================
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
                            [F3] Save   [F2] Toggle Debug  [ESC] Emergency Stop

"""

# =============================
# GLOBAL SETTINGS
# =============================
DEBUG_VIS = True              # F2로 토글
SHOW_INFER_MS = True          # 추론 시간 출력(간헐)
PRINT_EVERY_N = 10            # 로그 스팸 방지
TARGET_FPS = 20               # 프레임 제한(대상 FPS)
MOVE_DUR = 0.08               # 마우스 이동 시간(부드럽게)
ANTI_LAG_INTERVAL = 60        # 초. 60초마다 우클릭 유지로 렉 방지
ANTI_LAG_HOLD = 2.0           # 초. 우클릭 유지 시간

# 클래스 ID는 학습한 모델 순서를 그대로 사용
# 0: corpse, 1: monster, 2: ogre, 3: player
CLS_CORPSE, CLS_MONSTER, CLS_OGRE, CLS_PLAYER = 0, 1, 2, 3

# 스킬/행동 키 설정
ATTACK_KEY = "4"
TELEPORT_KEY = "0"
SKILL_KEYS = {5:120, 6:60, 7:60, 8:60, 9:60}  # 각 스킬 쿨다운(초)
ANY_SKILL_COOLDOWN = 10                        # 모든 스킬 공용 딜레이(초)

# =============================
# INIT
# =============================
os.system("cls" if os.name == "nt" else "clear")
print("gpu detected.." if torch.cuda.is_available() else "", ascii_art_load)

torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

config = Config.load()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('./train/weights/best.pt').to(device=device)
try:
    model.fuse()  # Conv+BN fuse (가능한 경우)
except Exception:
    pass

USE_HALF = bool(device == 'cuda')

# 더미로 워밍업
_dummy = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8)
_ = model.predict(_dummy, imgsz=config.IMG_SIZE, conf=0.25, verbose=False, half=USE_HALF)
cls_names = model.names

last_skill_use = {k: 0.0 for k in SKILL_KEYS}
last_any_skill_use = 0.0

pdi.PAUSE = 0
pdi.FAILSAFE = True  # 모서리로 마우스 이동 시 중단

os.system("cls" if os.name == "nt" else "clear")
print("gpu detected.." if torch.cuda.is_available() else "", ascii_art)

# =============================
# UTILS
# =============================
def now():
    return time.time()


def is_skill_ready(key:int) -> bool:
    return (now() - last_any_skill_use) >= ANY_SKILL_COOLDOWN and (now() - last_skill_use[key]) >= SKILL_KEYS[key]


def press_key(key:str, times:int=1, gap:float=0.06):
    key = str(key)
    for _ in range(times):
        pdi.keyDown(key)
        time.sleep(0.04)
        pdi.keyUp(key)
        time.sleep(gap)


def use_skill(key:int, burst:int=3):
    global last_any_skill_use
    press_key(str(key), times=burst, gap=0.05)
    last_skill_use[key] = now()
    last_any_skill_use = now()
    print(f"[스킬] {key}번 사용")


def get_screen_boxes(result_boxes, region):
    """Ultralytics Boxes -> 화면 좌표로 변환 + 필터링.
       corpse(0) 제거, player(3)는 최고 신뢰도 1개만 유지.
    """
    xyxy = result_boxes.xyxy.cpu().numpy() if hasattr(result_boxes.xyxy, 'cpu') else result_boxes.xyxy
    cls  = result_boxes.cls.cpu().numpy()  if hasattr(result_boxes.cls, 'cpu')  else result_boxes.cls
    conf = result_boxes.conf.cpu().numpy() if hasattr(result_boxes.conf, 'cpu') else result_boxes.conf

    screen_boxes = []
    x_off, y_off = region[0], region[1]

    for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
        c = int(c)
        if c == CLS_CORPSE:
            continue
        sx1, sy1, sx2, sy2 = int(x1 + x_off), int(y1 + y_off), int(x2 + x_off), int(y2 + y_off)
        cx, cy = (sx1 + sx2) // 2, (sy1 + sy2) // 2
        screen_boxes.append({
            "xyxy_screen": (sx1, sy1, sx2, sy2),
            "center_screen": (cx, cy),
            "cls": c,
            "conf": float(p),
        })

    # player(3) 박스는 최고 신뢰도 하나만 남김
    players = [b for b in screen_boxes if b["cls"] == CLS_PLAYER]
    if players:
        best_player = max(players, key=lambda b: b["conf"])
        screen_boxes = [b for b in screen_boxes if b["cls"] != CLS_PLAYER] + [best_player]

    return screen_boxes


def split_cluster_and_singles_sklearn(
    screen_boxes,
    target_cls=CLS_OGRE,
    min_cluster_size=4,
    max_k=3,
    max_cluster_radius=90, 
):
    targets = [b for b in screen_boxes if b["cls"] == target_cls]
    if not targets:
        return [], []

    pts  = np.array([b["center_screen"] for b in targets], dtype=np.float32)
    conf = np.array([b["conf"]           for b in targets], dtype=np.float32)
    N = len(pts)

    if N < min_cluster_size:
        order = np.argsort(-conf)
        return [], [tuple(map(int, pts[i])) for i in order]

    est_k = int(np.ceil(N / float(min_cluster_size)))
    k = max(2, min(max_k, est_k, N))

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    kmeans.fit(pts, sample_weight=conf)
    labels = kmeans.labels_

    cluster_targets = []
    single_indices  = []

    for i in range(k):
        idxs = np.where(labels == i)[0]
        if len(idxs) >= min_cluster_size:
            w = conf[idxs]
            wsum = float(w.sum())
            cx, cy = (pts[idxs] * w[:, None]).sum(axis=0) / max(wsum, 1e-6)

            # 가중치 RMS 반경 계산
            dx = pts[idxs, 0] - cx
            dy = pts[idxs, 1] - cy
            r_rms = float(np.sqrt(((dx*dx + dy*dy) * w).sum() / max(wsum, 1e-6)))

            if r_rms <= max_cluster_radius:  # ← 반경이 작을 때만 ‘진짜 군집’
                cluster_targets.append((wsum, int(cx), int(cy)))
            else:
                # 퍼진 군집은 단일 타깃으로 편입
                single_indices.extend(idxs.tolist())
        else:
            single_indices.extend(idxs.tolist())

    cluster_targets.sort(key=lambda t: t[0], reverse=True)
    single_indices.sort(key=lambda i: conf[i], reverse=True)

    clusters = [(x, y) for _, x, y in cluster_targets]
    singles  = [tuple(map(int, pts[i])) for i in single_indices]
    return clusters, singles



def nearby_units(center, screen_boxes, target_classes=(CLS_MONSTER, CLS_OGRE), radius=150):
    cx, cy = center
    res = []
    for b in screen_boxes:
        if b["cls"] in target_classes:
            bx, by = b["center_screen"]
            if math.hypot(bx - cx, by - cy) <= radius:
                res.append(b)
    return res

# =============================
# CORE LOOP
# =============================

def auto_hunt():
    if not config.WINDOW_TITLE:
        print("창을 선택해주세요")
        time.sleep(2)
        return

    frame_cnt = 0
    no_cnt = 0

    last_anti_lag = now()

    # 디버그 창 준비
    win_name = "result"
    created_window = False

    try:
        while True:
            loop_start = time.perf_counter()

            # 비상 정지
            if kb.is_pressed("esc"):
                print("[중단] ESC")
                break
            if kb.is_pressed("pagedown"):
                return

            # 주기적 렉 방지(우클릭 유지)
            if now() - last_anti_lag > ANTI_LAG_INTERVAL:
                frame_cnt = 0
                print("[렉 방지] 우클릭 유지")
                pdi.mouseDown(button="right")
                time.sleep(ANTI_LAG_HOLD)
                pdi.mouseUp(button="right")
                last_anti_lag = now()

            # 캡처
            img, region = get_window_img(config.WINDOW_TITLE)
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 버프/스킬 체크(예: 이미지로 상태 확인)
            try:
                pag.locate("sinsung.png", img, confidence=0.8)
            except ImageNotFoundException:
                print("신성 사용")
                press_key("3")

            # 추론
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            results = model.predict(
                frame,
                imgsz=config.IMG_SIZE,
                conf=config.CONFIDENCE,
                iou=0.45,
                half=USE_HALF,
                verbose=False,
                workers=0,
            )
            if device == 'cuda':
                torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - t0) * 1000.0
            if SHOW_INFER_MS:
                print(f"추론시간: {infer_ms:.1f} ms")

            result = results[0]

            # 디버그 시각화
            if DEBUG_VIS:
                try:
                    if not created_window:
                        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                        created_window = True
                    cv2.imshow(win_name, result.plot())
                    cv2.waitKey(1)
                except Exception:
                    pass

            # 박스 정리(화면 좌표)
            screen_boxes = get_screen_boxes(result.boxes, region)

            monsters = [b for b in screen_boxes if b["cls"] == CLS_MONSTER]
            ogres    = [b for b in screen_boxes if b["cls"] == CLS_OGRE]
            player   = next((b for b in screen_boxes if b["cls"] == CLS_PLAYER), None)

            # 주기적 아이템 인식 방지
            if len(monsters) + len(ogres) < 3:
                frame_cnt += 1
            else:
                frame_cnt = 0  # 몬스터가 충분히 있으면 카운트 초기화

            if frame_cnt > 30:
                frame_cnt = 0
                print("[처리] 아이템 인식 방지")
                press_key(TELEPORT_KEY)
                press_key(TELEPORT_KEY)
                press_key(TELEPORT_KEY)
                time.sleep(1)

            # 몬스터 없음 처리
            if not monsters and not ogres:
                no_cnt += 1
                if no_cnt >= 10:
                    frame_cnt = 0
                    no_cnt = 0
                    print("몬스터 없음 → 텔레포트")
                    press_key(TELEPORT_KEY)
                    time.sleep(1)
                    _fps_sleep(loop_start)
                    continue
                else:
                    print("몬스터 없음 → 대기")
                    cx = (region[0] + region[2]) // 2
                    cy = (region[1] + region[3]) // 2
                    pdi.moveTo(region[0] + 100, region[1] + 100)
                    _fps_sleep(loop_start)
                    continue
            else:
                no_cnt = 0

            # ============ 우선순위 1: 오우거 군집 처리 ============
            clusters, singles = split_cluster_and_singles_sklearn(
                screen_boxes,
                target_cls=CLS_OGRE,
                min_cluster_size=3,
                max_k=3,
                max_cluster_radius=70,
            )
            for cx, cy in clusters:
                pdi.moveTo(cx, cy, duration=MOVE_DUR)
                print("[군집] 오우거 군집 타격")
                time.sleep(1.7)
                if   is_skill_ready(7):
                    use_skill(7)
                elif is_skill_ready(8):
                    use_skill(8)
                else:
                    press_key(ATTACK_KEY)
                    print("쿨타임 중 → 평타")

            for cx, cy in singles:
                pdi.moveTo(cx, cy, duration=MOVE_DUR)
                print("[개별] 오우거 단일 타격")
                press_key(ATTACK_KEY)

            # ============ 우선순위 2: 플레이어 주변 광역 ============
            center = player["center_screen"] if player else (
                (region[0] + region[2]) // 2,
                (region[1] + region[3]) // 2,
            )
            units = nearby_units(center, screen_boxes, target_classes=(CLS_MONSTER, CLS_OGRE), radius=config.PLAYER_RADIUS)
            used = False
            if len(units) >= 3:
                print("[광역] 주변 3마리 이상 → 광역 스킬")
                pdi.moveTo(*center, duration=MOVE_DUR)
                time.sleep(1.7)
                if   is_skill_ready(5):
                    use_skill(5); used = True
                elif is_skill_ready(6):
                    use_skill(6); used = True
                elif is_skill_ready(9):
                    use_skill(9); used = True
                else:
                    print("광역 스킬 쿨타임")

            # ============ 우선순위 3: 일반 몬스터 처리 ============
            for box in monsters:
                if used and box in units:
                    continue  # 방금 광역으로 친 대상은 스킵
                cx, cy = box["center_screen"]
                label = cls_names.get(box["cls"], str(box["cls"])) if isinstance(cls_names, dict) else str(box["cls"]) 
                pdi.moveTo(cx, cy, duration=MOVE_DUR)
                print(f"[타격] {label} conf={box['conf']:.2f}")
                press_key(ATTACK_KEY)

            _fps_sleep(loop_start)

    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def _fps_sleep(loop_start):
    # 목표 FPS에 맞춰 슬립
    target_dt = 1.0 / max(1, TARGET_FPS)
    elapsed = time.perf_counter() - loop_start
    if elapsed < target_dt:
        time.sleep(target_dt - elapsed)

# =============================
# MAIN HOTKEY LOOP
# =============================

while True:
    try:
        if kb.is_pressed("f2"):
            DEBUG_VIS = not DEBUG_VIS
            print(f"DEBUG_VIS = {DEBUG_VIS}")
            time.sleep(0.25)

        if kb.is_pressed("pageup"):
            config.WINDOW_TITLE = get_window_tilte()
            print("\n선택됨:", config.WINDOW_TITLE)
            time.sleep(0.7)
            os.system("cls" if os.name == "nt" else "clear")
            print(ascii_art)

        elif kb.is_pressed("pagedown"):
            print("3초 후 자동사냥 시작")
            time.sleep(3)
            auto_hunt()
            os.system("cls" if os.name == "nt" else "clear")
            print(ascii_art)
            time.sleep(1.0)

        elif kb.is_pressed("f3"):
            config.save()
            print("저장 완료..")
            time.sleep(0.5)
            os.system("cls" if os.name == "nt" else "clear")
            print(ascii_art)

        time.sleep(0.03)
    except KeyboardInterrupt:
        print("[중단] KeyboardInterrupt")
        break
    except Exception as e:
        # 에러로 전체 중단되는 것 방지
        print("[오류]", e)
        time.sleep(0.5)
