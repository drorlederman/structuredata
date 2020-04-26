#save_images(fm.imgs, labels)
from skimage import io
import numpy as np

def read_images(file_name):
    imgs = np.load(file_name)
    labels = np.load(file_name + "_labels")
    return imgs, labels

def save_images(file_name, imgs, labels):
    np.save(file_name, imgs)
    np.save(file_name + "_labels", labels)
    #for i in range(0, imgs.shape[0]):
        #img_array = (img_array.flatten())
        #full_path = folder + str(labels.loc[i,'dec']) + "\\"
        #file_name = full_path + str(i) + '.png'
    #    io.imsave(file_name, imgs[i,:,:,:])

    return


