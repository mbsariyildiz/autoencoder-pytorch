from os.path import join, exists
from os import makedirs
from argparse import Namespace
import shutil

def model_info(model, model_name, log_dir):
  print ('Number of {} parameters: {}'.format(
    model_name,
    sum([p.data.nelement() for p in model.parameters()]))) 

  if log_dir:
    save_model_desc(
      model, join(log_dir, '{}_desc.txt'.format(model_name)))

def save_model_desc(model, path):
  with open(path, 'w') as fid:
    fid.write(str(model))

class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.__sum = 0
    self.__count = 0

  def update(self, val, n=1):
    self.__sum += val * n
    self.__count += n

  @property
  def avg(self):
    if self.__count == 0:
      return 0.
    return self.__sum / self.__count

def prepare_directory(directory):
  if exists(directory):
    print ('directory: %s already exists, backing up this folder ... ' % directory)
    
    backup_dir = directory + '_backup'
    if exists(backup_dir):
      print ('backup directory also exists, removing the backup directory first')
      shutil.rmtree(backup_dir, True)

    shutil.copytree(directory, backup_dir)
    shutil.rmtree(directory, True)

  makedirs(directory)

def write_logs(FLAGS):
  # save all setup into a log file
  if isinstance(FLAGS, Namespace):
    _dict = vars(FLAGS)
    _list = sorted(_dict.keys())
  else:
    _dict = FLAGS.__flags
    _list = sorted(_dict.keys())

  fid_setup = open(join(FLAGS.exp_dir, 'setup.txt'), 'w')
  for _k in _list:
    fid_setup.write('%s: %s\n' % (_k, _dict[_k]))
  fid_setup.flush()
  fid_setup.close()