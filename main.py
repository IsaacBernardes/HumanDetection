import os
from preprocessor import detect_borders


def main():

    work_dir = "images\\1"
    output_dir = "output"

    for file_name in os.listdir(work_dir):
        print("Working on file '{}'...".format(os.path.join(work_dir, file_name)))
        detect_borders(
            os.path.join(work_dir, file_name),
            os.path.join(output_dir, file_name)
        )


if __name__ == "__main__":
    main()
