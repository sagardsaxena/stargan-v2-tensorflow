import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--model', type=str)
args = parser.parse_args()

import os
from tqdm import tqdm

ckpts = os.listdir('checkpoint/{}/StarGAN_v2_{}_gan-gp/'.format(args.model, args.dataset))
ckpts = set([int(e.split('-')[1].split('.')[0]) for e in ckpts if '-' in e])
ckpts = sorted(ckpts)

for ckpt in tqdm(ckpts):
	cmd = "python3 main.py --dataset {} --phase test --max_to_keep -1 --checkpoint_dir checkpoint/{}/ --result_dir results/{}/ --log_dir logs/{}/ --sample_dir samples/{}/ --batch_size 4 --iteration 100000 --save_freq 5000 --augment_flag False --checkpoint {}".format(args.dataset, args.model, args.model, args.model, args.model, ckpt)
	os.system(cmd)

