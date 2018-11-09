import cv2
import numpy as np

ESC_KEY = 27
INTERVAL = 33
FRAME_RATE = 60

WINDOW_SRC = "src"
WINDOW_DIFF = "diff"
WINDOW_WB = "WhiteBlack"
WINDOW_CUTBK = "Cut background"

FILE_ORG = "ball2-1.mov"


def search_neighbor(p0, ps):
    L = np.array([])
    for i in range(ps.shape[0]):
        L = np.append(L, np.linalg.norm(ps[i]-p0))
        L[np.where(L == 0)] = 999  # 値が0の要素を無理やり書き換える
    return np.argmin(L), L[np.argmin(L)]  # 返り値は近傍点の(ラベル番号),(距離)


#  ウィンドウ命名
cv2.namedWindow(WINDOW_SRC)
cv2.namedWindow(WINDOW_DIFF)
cv2.namedWindow(WINDOW_WB)

# 元d
mov_org = cv2.VideoCapture(FILE_ORG)

# 最初のフレーム読み込み
has_next, i_frame = mov_org.read()

# 背景フレーム
back_frame = np.zeros_like(i_frame, np.float32)

time = 0  # 経過時間(フレーム)
while has_next:

    time += 1
    # 入力画像を浮動小数点型に変換
    f_frame = i_frame.astype(np.float32)

    # 差分
    diff_frame = cv2.absdiff(f_frame, back_frame)

    # 2値化
    gray_frame = cv2.cvtColor(diff_frame, cv2.COLOR_BGR2GRAY)
    thresh = 100  # しきい値
    max_pixel = 255
    ret, dst_frame = cv2.threshold(
        gray_frame,
        thresh,
        max_pixel,
        cv2.THRESH_BINARY
        )

    stats = centroids = near_index = near_dist = []

    # ラベリングのため深度変更
    dst_frame = np.uint8(dst_frame)

    if time >= 50:
        # ラベリング処理
        nlabels, labeledimg, stats, CoGs = \
            cv2.connectedComponentsWithStats(dst_frame)  # \は無視した上で処理される
        region = np.empty((0, 5), int)
        center = np.empty((0, 2), int)

        for n in range(nlabels):
            x, y, w, h, size = stats[n]

            if size > 30 and size < 1000:
                # 十分な大きさの領域のみ抽出
                region = np.vstack((region, stats[n]))
                center = np.vstack((center, CoGs[n]))

        for i in range(len(region)):
            x, y, w, h, size = region[i]
            near_index, near_dist = search_neighbor(center[i], center)
            # if time == 50:
            #    print("label" + str(i) + "\n near_index: " \
            # + str(near_index) +  " near_dist: " + str(near_dist))
            if near_dist <= 30:  # もし近くに矩形があれば
                xe, ye, we, he, se = region[near_index]
                mx1 = x; my1 = y; mx2 = x+w; my2 = y+h
                ex1 = xe; ey1 = ye; ex2 = xe + we; ey2 = ye + he
                if mx1 <= ex2 or ex1 <= mx2 or my1 <= ey2 or ey1 <= my2:
                    # 2つの重なった矩形を結合。4隅の点のうち結合した矩形の頂点になるものを選定
                    m1 = (mx1, my1); m2 = (mx2, my2)
                    e1 = (ex1, ey1); e2 = (ex2, ey2)
                    points = (m1, m2, e1, e2)
                    lm1 = np.linalg.norm(m1); lm2 = np.linalg.norm(m2)
                    le1 = np.linalg.norm(e1); le2 = np.linalg.norm(e2)
                    length = [lm1, lm2, le1, le2]
                    p1 = points[np.argmin(length)]
                    p2 = points[np.argmax(length)]
                    cv2.rectangle(
                        dst_frame, p1, p2, (255, 255, 255), 2)
                    if time == 50:
                        print("lect[" + str(i) + "]: " + str(p1) + ", " + str(p2) + " (combined with rect[" + str(near_index) + "])")
                else:  # もし近くに矩形があっても重ならなければ
                    cv2.rectangle(
                        dst_frame, (x-2, y-2), (x+w+2, y+h+2), (255, 255, 255), 2)
                    if time == 50:
                        print("lect[" + str(i) + "]: (" + str(x-2) + "," + str(y-2) + "), (" + str(x+w+2) + "," + str(y+h+2) + ")")
            else:  # もし近くに矩形がなければ
                cv2.rectangle(
                    dst_frame, (x-2, y-2), (x+w+2, y+h+2), (255, 255, 255), 2)
                if time == 50:
                    print("lect[" + str(i) + "]: (" + str(x-2) + "," + str(y-2) + "), (" + str(x+w+2) + "," + str(y+h+2) + ")")

    # 背景の更新
    cv2.accumulateWeighted(f_frame, back_frame, 0.025)

    # フレーム表示
    cv2.imshow(WINDOW_SRC, i_frame)
    cv2.imshow(WINDOW_DIFF, diff_frame.astype(np.uint8))
    cv2.imshow(WINDOW_WB, dst_frame)

    # Escキーで終了
    key = cv2.waitKey(INTERVAL)
    if key == ESC_KEY:
        break

    # 次のフレーム読み込み
    has_next, i_frame = mov_org.read()

# 終了処理
cv2.destroyAllWindows()
mov_org.release()
