import pandas as pd
import argparse
import torchvision.transforms as tfm
from torch.utils.data import DataLoader
from dataset import ChestDataset
from utils import Trainer
from model import AutoEncoder

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--debug', action='store_true', default=False, help='debug')
	parser.add_argument('--datapath', type=str, help="path of the dataset", required=True)
	parser.add_argument('--ckpt_path', type=str, help="path to save checkpoint", required=True)
	parser.add_argument('--stage', type=str, help="stage of the model", required=True)
	parser.add_argument('--max_epochs', type=int, default = 10, help="total epochs", required=False)
	parser.add_argument('--batch_size', type=int, default = 32, help="batch size", required=False)
	parser.add_argument('--learning_rate', type=int, default = 5e-4, help="learning rate", required=False)
	parser.add_argument('--grad_norm_clip', type=int, default = 1.0, help="clip gradients at this value", required=False)
	parser.add_argument('--num_workers', type=int, default = 4, help="number of workers", required=False)


	args = parser.parse_args()

	data = pd.read_csv(args.datapath + '.csv')

	train_data = data[data['split']=='train'].reset_index(drop=True)
	val_data = data[data['split']=='val'].reset_index(drop=True)

	if args.debug:
		train_data = train_data.sample(3*args.batch_size).reset_index(drop=True)
		val_data = val_data.sample(3*args.batch_size).reset_index(drop=True)

	train_tfm = tfm.Compose([tfm.ToPILImage(),
	                         tfm.RandomCrop(CROP_SIZE),
	                         tfm.RandomHorizontalFlip(),
	                         tfm.RandomRotation(degrees=5),
	                         tfm.ColorJitter(contrast=0.25),
	                         tfm.ToTensor(),
	                         tfm.Normalize(mean=[0.485, 0.456, 0.406],
	                                     std=[0.229, 0.224, 0.225])])

	val_tfm = tfm.Compose([tfm.ToTensor(),
	                       tfm.Normalize(mean=[0.485, 0.456, 0.406],
	                                     std=[0.229, 0.224, 0.225])])

	train_dataset = ChestDataset(train_data, train_tfm, mode='train')
	val_dataset = ChestDataset(val_data, val_tfm, mode='val')

	trainloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
	valloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

	if args.stage == 'stage1':
		model = DenseNet121(classCount = 14, isTrained = False)
	elif args.stage == 'stage2':
		model = AutoEncoder()
	else:
		model = AECNN(classCount = 14, args = args)
		model.load_state_dict(torch.load(args.ckpt_path), strict = False)

	trainer = Trainer(model, trainloader, valloader, args)
	trainer.train()


