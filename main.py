import os
import shutil
import cv2
from preprocessor import sliding_image
from model import Classifier


def main():

    work_dir = "images\\1"

    if os.path.exists("output"):
        shutil.rmtree("output", ignore_errors=False, onerror=None)

    os.mkdir("output")

    for index, file_name in enumerate(os.listdir(work_dir)):

        print("Working on file '{}'...".format(os.path.join(work_dir, file_name)))
        sliding_image(
            os.path.join(work_dir, file_name),
            file_name,
            3
        )

        if index > 100:
            break

    cf = Classifier()
    for index, file_name in enumerate(os.listdir("output")):
        file_path = os.path.join("output", file_name)
        img = cv2.imread(file_path)
        persons, frame = cf.detect(img)
        print("Classifying {}".format(file_path))

        if persons > 0:
            cv2.imwrite(file_path, frame)
        else:
            os.remove(file_path)


if __name__ == "__main__":
    main()
