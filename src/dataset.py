import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

np.random.seed(67)

class celebA(object):
  def __init__(self, data_dir, red_rate=0.0, test_split=0.2, validation_split=0.0):
    self.data_dir = data_dir
    self.image_files = glob.glob(os.path.join(data_dir, '*.jpg'))
    self.image_files = np.asarray(self.image_files)
    print ('{} images found for celebA'.format(self.image_files.shape[0]))

    n_all = self.image_files.shape[0]
    if red_rate > 0.0:
      n_all = int(n_all * (1.0 - red_rate))

    n_train = int(n_all * (1.0 - test_split))
    order = np.random.permutation(n_all)

    self.train_inds = order[:n_train]
    self.test_inds = order[n_train:]

    if validation_split > 0.0:
      n_train = int(n_train * (1.0 - validation_split))
      self.test_inds = self.train_inds[n_train:]
      self.train_inds = self.train_inds[:n_train]

    self.train_images = self.image_files[self.train_inds]
    self.test_images = self.image_files[self.test_inds]
    print ('number of training images {}'.format(self.train_images.shape[0]))
    print ('number of test images {}'.format(self.test_images.shape[0]))

class celebA_Subset(Dataset):
  def __init__(self, image_files, image_transform):
    self.image_files = image_files
    self.image_transform = image_transform
    
  def __len__(self):
    return self.image_files.shape[0]

  def __getitem__(self, index):
    file_name = self.image_files[index]
    image = Image.open(file_name)
    image = self.image_transform(image)
    return image