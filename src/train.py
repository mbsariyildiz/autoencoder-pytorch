import os
import argparse
import shutil
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import config
import utils
from tqdm import trange
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tensorboard_logger import configure, log_value

def save_checkpoint(state, dir, is_best=False):
  ckpt_file = os.path.join(dir, 'model.ckpt')
  torch.save(state, ckpt_file)
  if is_best:
    shutil.copyfile(
      ckpt_file, 
      os.path.join(dir, 'model_best.ckpt'))

def train_epoch(enc, dec, enc_optim, dec_optim, criteria, data_loader, device):
  train_loss = utils.AverageMeter()
  enc.train()
  dec.train()

  for x in data_loader:
    enc.zero_grad()
    dec.zero_grad()

    x = x.to(device)
    z = enc(x)
    x_hat = dec(z)

    loss = criteria(x_hat, x)
    loss.backward()
    enc_optim.step()
    dec_optim.step()

    train_loss.update(loss.item(), x.size(0))

  return train_loss.avg

def test(enc, dec, criteria, data_loader, device):
  test_loss = utils.AverageMeter()
  enc.eval()
  dec.eval()

  for x in data_loader:
    x = x.to(device)
    z = enc(x)
    x_hat = dec(z)

    loss = criteria(x_hat, x)
    test_loss.update(loss.item(), x.size(0))

  return test_loss.avg

def main(args):

  train_set, test_set = config.load_dataset(args)

  train_loader = DataLoader(train_set, args.batch_size, 
    shuffle=True, num_workers=16, pin_memory=True)
  test_loader = DataLoader(test_set, args.batch_size, 
    shuffle=False, num_workers=8, pin_memory=True)

  enc, dec = config.load_model(args)
  enc_optim = torch.optim.Adam(enc.parameters(), 1e-3)
  dec_optim = torch.optim.Adam(dec.parameters(), 1e-3)
  enc_optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(enc_optim, [100, 150, 200], 0.5)
  dec_optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(dec_optim, [100, 150, 200], 0.5)
  starting_epoch = 0
  
  if args.ckpt_file:
    print ('loading checkpoint file ... ')
    if args.device == 'cpu':
      ckpt = torch.load(args.ckpt_file, map_location=lambda storage, loc: storage)
    else:
      ckpt = torch.load(args.ckpt_file)

    enc.load_state_dict(ckpt['encoder'])
    dec.load_state_dict(ckpt['decoder'])
    enc_optim.load_state_dict(ckpt['encoder_optim'])
    dec_optim.load_state_dict(ckpt['decoder_optim'])
    starting_epoch = ckpt['epoch']

  train_criteria = torch.nn.MSELoss()
  test_criteria = torch.nn.L1Loss()

  min_test_loss = 1e5

  for epoch in trange(starting_epoch, args.n_epochs, ncols=100):

    train_loss = train_epoch(
      enc, dec, enc_optim, dec_optim, train_criteria, train_loader, args.device)
    test_loss = test(
      enc, dec, test_criteria, test_loader, args.device)
    enc_optim_scheduler.step(test_loss)
    dec_optim_scheduler.step(test_loss)

    log_value('train_loss', train_loss, epoch)
    log_value('test_loss', test_loss, epoch)

    enc.eval(); dec.eval()
    for _setname, _set in zip(['train', 'test'], [train_set, test_set]):
      n_compare = 64
      inds = np.random.choice(len(_set), n_compare)
      x_comb = []
      for i in inds:
        x_comb.append(_set[i].unsqueeze(0))
        x_comb.append(
          dec(
            enc(x_comb[-1].to(args.device))
          ).detach().to('cpu'))

      save_image(
        torch.cat(x_comb, 0), 
        os.path.join(args.exp_dir, 'real_vs_reconstructed_{}.png'.format(_setname)),
        2)

    save_checkpoint(
      {
        'epoch': epoch,
        'encoder': enc.state_dict(), 
        'decoder': dec.state_dict(),
        'encoder_optim': enc_optim.state_dict(),
        'decoder_optim': dec_optim.state_dict(),
        'test_loss': test_loss
      },
      args.exp_dir, 
      test_loss < min_test_loss)

    if test_loss < min_test_loss:
      min_test_loss = test_loss

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='')
  parser.add_argument('--exp_dir', type=str, default='')
  parser.add_argument('--ckpt_file', type=str, default='')
  parser.add_argument('--device', type=str, default='cuda')
  parser.add_argument('--multi_gpu', action='store_true')
  parser.add_argument('--dataset', type=str, default='celebA')
  parser.add_argument('--test_split', type=float, default=0.2)
  parser.add_argument('--red_rate', type=float, default=0.0)
  parser.add_argument('--validation_split', type=float, default=0.0)
  parser.add_argument('--d_latent', type=int, default=128)
  parser.add_argument('--batch_size', type=int, default=128)
  parser.add_argument('--n_epochs', type=int, default=150)
  args = parser.parse_args()

  if args.device == 'cuda' and torch.cuda.is_available():
    from subprocess import call
    print ('available gpus:')
    call(["nvidia-smi", 
           "--format=csv", 
           "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    cudnn.benchmark = True
  else:
    args.device = 'cpu'

  utils.prepare_directory(args.exp_dir)
  utils.write_logs(args)
  configure(args.exp_dir)
  main(args)