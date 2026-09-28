import pygetwindow as gw
import numpy as np
import mss
import cv2

supported_games = ["Minecraft", "Roblox"]
_sct = mss.MSS()

def find_game_window():
    windows = gw.getAllWindows()
    
    for game in supported_games:
        for wndw in windows:
            if game.lower() in wndw.title.lower():
                return wndw
    return None

def capture_game(wndw):
        regions = {
            "top": wndw.top,
            "left": wndw.left,
            "width": wndw.width,
            "height": wndw.height
        }

        screenshot = _sct.grab(regions)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        return img

def grab_features(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges =cv2.Canny(grayscale, 100, 200)
    return edges


# test function created by ai
win = find_game_window()
if win:
    frame = capture_game(win)
    edges = grab_features(frame)
    print(f"Frame shape: {frame.shape}")
    print(f"Edge pixels detected: {np.sum(edges > 0)}")  # Should be > 0, not 0