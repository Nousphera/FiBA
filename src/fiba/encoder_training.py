"""Encoder/Decoder training with explicit configuration.
Modified 2026-10-15 from fl_battacks/encoder/src/main.py; see PROVENANCE.md.
"""
import argparse
import itertools
import json
from pathlib import Path
import random
import numpy as np
import torch
import torchvision
from .encoder_poisoning import Encoder, NoiseDataset

class Decoder(torch.nn.Module):

	def __init__(self):
		super(Decoder, self).__init__()
		self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
		self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, padding=1)
		self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
		self.conv4 = torch.nn.Conv2d(32, 16, kernel_size=3, padding=1)
		self.conv5 = torch.nn.Conv2d(16, 8, kernel_size=3, padding=1)
		self.conv6 = torch.nn.Conv2d(8, 1, kernel_size=3, padding=1)
		self.ln1 = torch.nn.Linear(1024, 512)
		self.ln2 = torch.nn.Linear(512, 100)

	def forward(self, x):
		x = torch.nn.functional.relu(self.conv1(x))
		x = torch.nn.functional.relu(self.conv2(x))
		x = torch.nn.functional.relu(self.conv3(x))
		x = torch.nn.functional.relu(self.conv4(x))
		x = torch.nn.functional.relu(self.conv5(x))
		x = self.conv6(x)
		return x

def train(_encoder, _decoder, train_ds, test_ds, epochs=10, lr=5e-4, beta=0.7, freq=None, device='cuda'):

    total_loss = {'losses_train': []}
    optimizer_enc = torch.optim.AdamW(_encoder.parameters(), lr=lr)
    optimizer_dec = torch.optim.AdamW(_decoder.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    scheduler_enc = torch.optim.lr_scheduler.StepLR(optimizer_enc, step_size=5, gamma=0.5)
    scheduler_dec = torch.optim.lr_scheduler.StepLR(optimizer_dec, step_size=5, gamma=0.5)

    for epoch in range(epochs):
        epoch_enc_loss = 0
        epoch_dec_loss = 0
        for i, ((x, _), (_x, _)) in enumerate(zip(train_ds, itertools.cycle(test_ds))):
            optimizer_enc.zero_grad()
            optimizer_dec.zero_grad()
            x = x.to(device, non_blocking=True)
            _x = _x.expand(x.shape[0], -1, -1, -1).to(device, non_blocking=True)
            # Forward pass
            encoded = _encoder(x, _x)
            decoded = _decoder(encoded)
            # Compute losses
            enc_loss = loss_fn(encoded, x)
            dec_loss = enc_loss + beta * loss_fn(decoded, _x)
            # Backward pass and optimization
            dec_loss.backward()
            torch.nn.utils.clip_grad_norm_(_encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(_decoder.parameters(), max_norm=1.0)
            optimizer_enc.step()
            optimizer_dec.step()

            epoch_enc_loss += enc_loss.item()
            epoch_dec_loss += dec_loss.item()
            if freq is not None:
                if (i + 1) % freq == 0 or i == 0 or (i + 1) == len(train_ds):
                    print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_ds)}], '
                          f'Loss Encoder: {enc_loss.item():.4f}, Loss Decoder: {dec_loss.item():.4f}')
        scheduler_enc.step()
        scheduler_dec.step()
        epoch_enc_loss /= len(train_ds)
        epoch_dec_loss /= len(train_ds)
        total_loss['losses_train'].append({'enc_loss': epoch_enc_loss, 'dec_loss': epoch_dec_loss})
        print(f'Epoch [{epoch + 1}/{epochs}] Average Encoder Loss: {epoch_enc_loss:.4f}, Average Decoder Loss: {epoch_dec_loss:.4f}')
    return total_loss

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=Path('datasets'))
    p.add_argument('--download',action='store_true')
    p.add_argument('--synthetic',action='store_true',help='Four random images for a pipeline check only')
    p.add_argument('--output',type=Path,default=Path('encoder.pt'))
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--lr',type=float,default=5e-4)
    p.add_argument('--beta',type=float,default=.7)
    p.add_argument('--noise-dim',type=int,default=200)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cpu')
    args = p.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        p.error('epochs and batch size must be positive')
    if args.output.exists():
        p.error('output checkpoint already exists; choose another file')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.synthetic:
        dataset = torch.utils.data.TensorDataset(torch.rand(4,3,32,32),torch.zeros(4,dtype=torch.long))
    else:
        dataset = torchvision.datasets.CIFAR10(str(args.data_root),train=True,download=args.download,
            transform=torchvision.transforms.ToTensor())
    batch_size = min(args.batch_size,len(dataset))
    loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True,drop_last=True)
    noises = NoiseDataset(size=4 if args.synthetic else 10000,noise_dim=args.noise_dim,image_size=32)
    noise_loader = torch.utils.data.DataLoader(noises,batch_size=min(batch_size,len(noises)),shuffle=True,drop_last=True)
    encoder,decoder = Encoder().to(args.device),Decoder().to(args.device)
    history = train(encoder,decoder,loader,noise_loader,epochs=args.epochs,lr=args.lr,beta=args.beta,device=args.device)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(encoder.cpu().state_dict(),args.output)
    report = {'epochs':args.epochs,'synthetic':args.synthetic,'seed':args.seed,'noise_dim':args.noise_dim,'history':history}
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report

if __name__ == '__main__':
    main()
