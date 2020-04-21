import cv2

def get_ball_pos(index, first_run=False):
    if not first_run:
        img_ball = cv2.imread("Img/bg.jpg")
        first_run = True

    xc = 0
    yc = 0

    img_work = cv2.imread("Img/" + str(index) + ".jpg")
    cv2.imshow(img_work)

    fore =
