import cv2
import numpy as np

## インポートした画像を一つの文字ごとに切り分ける関数
def trimming(array, num, path):
    """画像を文字ごとに切り分ける

    Parameters:
    ----------
    array : int array
        前処理をした画像
    
    num : int
        トリミングの開始位置
    """

    outpath = path.replace(".png", "_") + str(num + 1) + ".png"
    startNum = num * 200
    array_list = []

    for i in range(200):
        array_list.append(array[i][startNum:(startNum + 200)])

    trimming_array = np.array(array_list)

    cv2.imwrite(outpath, trimming_array)

def createimg(path):
    ## 画像のインポート
    img_array = cv2.imread(path)              ## nparrayになっている
    height = img_array.shape[0]
    width = img_array.shape[1]

    img_array = np.array(img_array, dtype="uint8")

    ## 画像の前処理
    img_array = img_array.ravel()             ## 平坦化
    img_array = img_array[2::3]               ## 2から始めることで、黄色（枠の部分）を消す
    img_array = img_array.reshape(200, 1000)  ## reshapeで一次元配列を二次元配列にする

    ## cv2.imwrite(outpath, img_array)

    ## 文字ごとに取り出す
    trimming(img_array, 0, path)        ## 「あ」のデータ
    trimming(img_array, 1, path)        ## 「い」のデータ
    trimming(img_array, 2, path)        ## 「う」のデータ
    trimming(img_array, 3, path)        ## 「え」のデータ
    trimming(img_array, 4, path)        ## 「お」のデータ


def ondisplay(answer_array, path):
    import cv2
    import matplotlib.pyplot as plt

    print("★☆★☆★☆★☆ 判定結果 ☆★☆★☆★☆★\n")

    fig = plt.figure()

    x = 7
    y = 8

    for i in range(1, 6): ##1 = あ,2 = い,3 = う,4 = え,5 = お
        ## 手書き「あ」
        img1_path = path.replace(".png", "_") + str(i) + ".png"
        img1 = cv2.imread(img1_path)

        ## answer
        answer_name = str(answer_array[(i-1)])
        answer_name = answer_name.replace("['", "").replace('(', ' (').replace("'", "").replace("]", "")
        img2_path = "/content/drive/MyDrive/data/answerimg/" + answer_name + ".jpg"
        img2 = cv2.imread(img2_path)
        view_name = answer_name.replace(" (", "").replace("1", "").replace("2", "").replace("3", "").replace("4", "").replace("5", "").replace(")", "")

        if(i == 1):
            j = 1
        elif(i == 2):
            j = 6
        elif(i == 3):
            j = 25
        elif(i == 4):
            j = 30
        else:
            j = 49

        img1plot = j
        ax1 = fig.add_subplot(x, y, img1plot)
        ax1.set_title("Hand",fontsize=15)                 ##タイトルの設定
        plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        img2plot = (j + 2)
        ax2 = fig.add_subplot(x, y, img2plot)
        ax2.set_title(view_name,fontsize=15)                   ##タイトルの設定
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    count = 0

    for i in range(5):
        if str(i + 1) in str(answer_array[i]):
            count += 1

    print("全体正答率：" + str(count / 5 * 100) + " %\n")


def judge(path):
    from PIL import Image, ImageOps
    import numpy as np
    import joblib

    answer_array = []

    for i in range(1, 6):
        trimming_img_path = path.replace(".png", "_") + str(i) + ".png"
        model_path = "/content/drive/MyDrive/data/model/fontAll_GBMmodel_noparam.pkl"

        model = joblib.load(model_path)

        input_img = Image.open(trimming_img_path)
        input_img = ImageOps.invert(input_img)                      ## 色反転
        input_img = input_img.resize((40, 40), Image.LANCZOS)       ## リサイズはPILの方が精度が良い

        array = np.array(input_img, dtype="uint8")
        array = array.reshape((-1, 1600))

        answer_array.append(model.predict(array))

        # print(model.predict(array))

    ondisplay(answer_array, path)

    return answer_array

