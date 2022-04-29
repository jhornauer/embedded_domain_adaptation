import os
import pickle

if __name__ == '__main__':
    dir = os.path.join(os.getcwd(), 'Splits', 'vkitti')

    for file in os.listdir(dir):
        with open(os.path.join(dir, file), 'rb') as f:
            imgs = pickle.load(f)
        for i, item in enumerate(imgs):
            path = item[0]
            path = path.split('\\')
            # path = path[0].split('/') + [path[1], path[2]]
            # path = path[6] # os.path.join(path[4], path[5])
            path = os.path.join(path[0], path[1], path[2], path[3], path[4], path[5])
            item = (path, item[1])
            # path = os.path.join(path[1], path[2], path[3], path[4], path[5])
            # item = (path.replace('png', 'jpg'), item[1])
            imgs[i] = item
        with open(os.path.join(dir, file), 'wb') as f:
            pickle.dump(imgs, f)