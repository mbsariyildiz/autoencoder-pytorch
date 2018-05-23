import torch
import dataset
import encoder
import decoder
import torchvision
import torchvision.transforms as transforms

def load_dataset(args):

  if args.dataset == 'celebA':
    train_transform = transforms.Compose([
      transforms.Resize([64, 64]),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor()])
    test_transform = transforms.Compose([
      transforms.Resize([64, 64]),
      transforms.ToTensor()])

    celeba = dataset.celebA(
      args.data_dir, args.red_rate, args.test_split, args.validation_split)
    train_set = dataset.celebA_Subset(celeba.train_images, train_transform)
    test_set = dataset.celebA_Subset(celeba.test_images, test_transform)

  return train_set, test_set

def load_model(args):

  if args.dataset == 'celebA':
    enc = encoder.celebA_Encoder(args.d_latent, args.device, args.exp_dir)
    dec = decoder.celebA_Decoder(args.d_latent, args.device, args.exp_dir)

  if (args.device == 'cuda') and ('multi_gpu' in args) and (args.multi_gpu == True):
    print ('replicating the model on multiple gpus ... ')
    enc = torch.nn.DataParallel(enc)
    dec = torch.nn.DataParallel(dec)

  return enc, dec