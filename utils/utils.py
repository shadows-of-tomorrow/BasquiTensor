import os


def makedir(path, parent=0):
    if not os.path.exists(path):
        for k in range(parent):
            path = os.path.dirname(path)
        os.mkdir(path)
