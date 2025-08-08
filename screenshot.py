import pygetwindow as gw
from PIL import ImageGrab   


def get_window_tilte():
    titles = gw.getAllTitles()
    titles = [t for t in titles if t != ""]

    print("Metin 창을 선택하여주세요.")
    print("-"*50)

    for i, t in enumerate(titles):
        print(i+1, '.', t)

    print("-"*50)

    n=int(input("창 번호: "))

    selected_title=titles[n-1]
    return selected_title


def get_window_img(title):
    win = gw.getWindowsWithTitle(title)[0]

    x, y   = win.topleft       
    w, h   = win.size
    
    region=(x, y, x + w, y + h)

    img = ImageGrab.grab(bbox=region)

    return img, region