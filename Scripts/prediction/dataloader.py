import sys

from numpy import save

sys.path.append("..")

import numpy as np
from mutation.attributes import Attributes


class DataLoader:
    def __init__(self, calculated_csv_path, delete_sub_classes=False):
        self.read_csv(calculated_csv_path, delete_sub_classes)

    def read_csv(self, calculated_csv_path, delete_sub_classes):
        file = open(calculated_csv_path, "r")
        X_arr = []
        y_arr = []
        for line in file.readlines():
            attr = Attributes(line)
            if delete_sub_classes:
                if "$" in attr.class_name:
                    continue
            X_arr.append(attr.get_item()[0])
            if attr.get_item()[1] > 0:
                y_arr.append(1)
            else:
                y_arr.append(0)
        self.X = np.array(X_arr)
        self.y = np.array(y_arr)


class DataSaver:
    def save_data(save_path, X, y):
        line_count = 0
        lines = ""
        for xdx, x in enumerate(X):
            line = str(line_count) + ";"
            for idx, i in enumerate(x):
                line += str(i) + ";"
            line += str(y[xdx]) + "\n"
            lines += line

        f = open(save_path, "w")
        f.write(lines)
        f.close()


def create_merged_data_folder(data_csv_paths, save_path):
    save_lines = ""
    for calculated_csv_path in data_csv_paths:
        file = open(calculated_csv_path, "r")
        for line in file.readlines():
            if line == "\n":
                continue
            save_lines += line

    f = open(save_path, "w")
    f.write(save_lines)
    f.close()
