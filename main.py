import os
import shutil
import cv2
from preprocessor import sliding_image, improve_image


def main():

    work_dir = "images\\train\\images"

    if os.path.exists("output"):
        shutil.rmtree("output", ignore_errors=False, onerror=None)

    os.mkdir("output")

    for index, file_name in enumerate(os.listdir(work_dir)):

        img = cv2.imread(os.path.join(work_dir, file_name))
        results = improve_image(img)

        cv2.imshow("Original", img)

        for i in range(len(results)):
            cv2.imshow("CROP-" + str(i), results[i])

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # cv2.imshow("Janela", results)
        # cv2.waitKey(0)

        if index >= 100:
            break

    # cf = Classifier()
    # for index, file_name in enumerate(os.listdir("output")):
    #     file_path = os.path.join("output", file_name)
    #     img = cv2.imread(file_path)
    #     persons, frame = cf.detect(img)
    #     print("Classifying {}".format(file_path))
    #
    #     if persons > 0:
    #         cv2.imwrite(file_path, frame)
    #     else:
    #         os.remove(file_path)


def video():

    camera = cv2.VideoCapture(0)

    while True:

        ret, frame = camera.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        results = improve_image(frame)

        for i in range(len(results)):
            cv2.imshow("Janela" + str(i), results[i])

        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    main()
    # video()
