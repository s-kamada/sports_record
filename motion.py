import cv2
import numpy as np

# Esc キー
ESC_KEY = 0x1b
# s キー
S_KEY = 0x73
MAX_FEATURE_NUM = 2
# インターバル （1000 / フレームレート）
INTERVAL = 30

class Motion:
    # コンストラクタ
    def __init__(self):
        # 表示ウィンドウ
        cv2.namedWindow("motion")
        cv2.namedWindow("canny_edges")
        # マウスイベントのコールバック登録
        cv2.setMouseCallback("motion", self.onMouse)
        # 映像
        self.video = cv2.VideoCapture(3)
        self.interval = INTERVAL
        self.frame = None
        self.gray_next = None
        self.gray_prev = None
        self.features = None

    # メインループ
    def run(self):

        # 最初のフレームの処理
        end_flag, self.frame = self.video.read()
        output = self.frame

        while end_flag:

            self.gray_next = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            output = self.gray_next

            if self.features is not None:
                cv2.circle(self.frame, (self.features[0][0], self.features[0][1]), 2, (0, 0, 255), -1, 8, 0)
                if len(self.features) == 2:
                    cv2.rectangle(self.frame, (self.features[0][0], self.features[0][1]),(self.features[1][0], self.features[1][1]), (0, 0, 255), 2)
                    x = [int(self.features[0][0]), int(self.features[1][0])]
                    y = [int(self.features[0][1]), int(self.features[1][1])]
                    x.sort()
                    y.sort()
                    output = self.gray_next[y[0]:y[1],x[0]:x[1]]

            canny_edges = cv2.Canny(output,100,200)

            # 表示
            cv2.imshow("canny_edges",canny_edges)
            cv2.imshow("motion", self.frame)

            # 次のループ処理の準備
            self.gray_prev = self.gray_next
            end_flag, self.frame = self.video.read()
            if end_flag:
                self.gray_next = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            # インターバル
            key = cv2.waitKey(self.interval)
            # "Esc"キー押下で終了
            if key == ESC_KEY:
                break
            # "s"キー押下で一時停止
            elif key == S_KEY:
                self.interval = 0


        # 終了処理
        cv2.destroyAllWindows()
        self.video.release()


    def onMouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.features is None:
            self.addFeature(x, y)
            return
        else:
            self.addFeature(x, y)

        return


    def addFeature(self, x, y):
        if self.features is None:
            self.features = np.empty((0,2), int)
            self.features = np.append(self.features, np.array([[x, y]]), axis = 0).astype(np.float32)

        elif len(self.features) >= MAX_FEATURE_NUM:
            print("max feature num over: " + str(MAX_FEATURE_NUM))
            self.features = np.empty((0,2), int)
            self.features = np.append(self.features, np.array([[x, y]]), axis = 0).astype(np.float32)

        else:
            self.features = np.append(self.features, np.array([[x, y]]), axis = 0).astype(np.float32)



if __name__ == '__main__':
    Motion().run()
