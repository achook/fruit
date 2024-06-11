import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt

path_red = 'output/red/'
path_yellow = 'output/yellow/'


def label_apples(path):
    labels = []
    images = []
    path_unlabeled = os.path.join(path, 'unlabeled')
    path_correct = os.path.join(path, 'correct')
    path_incorrect = os.path.join(path, 'incorrect')
    print(f"Labeling images in {path_unlabeled}")
    print("Press '1' for correct, '0' for incorrect, 'q' to quit")


    for filename in os.listdir(path_unlabeled):
        img = cv2.imread(os.path.join(path_unlabeled, filename))
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()
        label = input('Label (1, 0): ')
        if label == 'q':
            break
        if label == '1':
            shutil.move(os.path.join(path_unlabeled, filename), path_correct)
        elif label == '0':
            shutil.move(os.path.join(path_unlabeled, filename), path_incorrect)
    #     images.append(img)
    #     labels.append(label)

    # images = np.asarray(images, dtype="object")
    # labels = np.array(labels, dtype="int")

    # np.save(f'./output/{path.split("/")[-2]}_images.npy', images, allow_pickle=True)
    # np.save(f'./output/{path.split("/")[-2]}_labels.npy', labels)

    return labels, images 


# red_labels, red_images = label_apples(path_red)
yellow_labels, yellow_images = label_apples(path_yellow)

# yellow_images= np.load('output/yellow_images.npy', allow_pickle=True)
# yellow_labels = np.load('output/yellow_labels.npy')


# red_labels = np.load('output/red_labels.npy')
# red_images = np.load('output/red_images.npy', allow_pickle=True)


# labels = np.concatenate((red_labels, yellow_labels))
# images = np.concatenate((red_images, yellow_images))

# assert len(images) == len(labels)

# print("saving data ...")

# np.save("./output/all_images.npy", images)
# np.save("./output/all_labels.npy", labels)

# print("done")