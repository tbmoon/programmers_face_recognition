import os
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from data_loader import FaceDataset, get_dataloader
from models import BaseModel, DenseCrossEntropy, ArcFaceLoss


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main(args):

    os.makedirs(os.path.join(os.getcwd(), 'logs'), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), 'models'), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), 'png'), exist_ok=True)

    data_loaders, data_sizes = get_dataloader(
        input_dir=args.input_dir,
        phases=['train', 'valid'],
        batch_size=args.batch_size,
        num_workers=args.num_workers)
    
    model = BaseModel(args.bn_mom, args.embed_size, args.num_classes)
    model = model.to(device)

    criterion = DenseCrossEntropy()
    criterion_arc = ArcFaceLoss()
    #criterion = nn.CrossEntropyLoss(reduction='sum')

    if args.load_model:
        checkpoint = torch.load(os.path.join(os.getcwd(), 'models/model.ckpt'))
        model.load_state_dict(checkpoint['state_dict'])
        
    params = list(model.parameters())
    optimizer = optim.Adam(params, lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    for epoch in range(args.num_epochs):

        for phase in ['train', 'valid']:
            since = time.time()
            running_loss = 0.0
            running_loss_arc = 0.0
            running_corrects = 0.0
            running_size = 0.0

            if phase == 'train':
                model.train()
            else:
                model.eval()

            for batch_idx, batch_sample in enumerate(data_loaders[phase]):
                optimizer.zero_grad()
                
                image = batch_sample['image']
                label = batch_sample['label']
                image = image.to(device)
                label = label.to(device)
                sample_size = image.size(0)
                
                with torch.set_grad_enabled(phase == 'train'):
                    total_loss = 0.0
                    output_arc, output = model(image)           # [batch_size, num_classes]

                    _, pred = torch.max(output, 1)
                    loss = criterion(output, label)
                    loss_arc = criterion(output_arc, label)
                    coeff = args.metric_loss_coeff
                    total_loss = (1 - coeff) * loss + coeff * loss_arc

                    if phase == 'train':
                        total_loss.backward()
                        _ = nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                        optimizer.step()
            
                running_loss += loss.item()
                running_loss_arc += loss_arc.item()
                running_corrects += torch.sum(pred == label)
                running_size += sample_size

            epoch_loss = running_loss / running_size
            epoch_loss_arc = running_loss_arc / running_size
            epoch_accuracy = float(running_corrects) / running_size

            print('| {} SET | Epoch [{:02d}/{:02d}]'.format(phase.upper(), epoch+1, args.num_epochs))
            print('\t*- Loss              : {:.4f}'.format(epoch_loss))
            print('\t*- Loss_Arc          : {:.4f}'.format(epoch_loss_arc))
            print('\t*- Accuracy          : {:.4f}'.format(epoch_accuracy))

            # Log the loss in an epoch.
            with open(os.path.join(os.getcwd(), 'logs/{}-log-epoch-{:02}.txt').format(phase, epoch+1), 'w') as f:
                f.write(str(epoch+1) + '\t' +
                        str(epoch_loss) + '\t' +
                        str(epoch_accuracy))

            # Save the model check points.
            if phase == 'train' and (epoch+1) % args.save_step == 0:
                torch.save({'epoch': epoch+1,
                            'state_dict': model.state_dict()},
                           os.path.join(os.getcwd(), 'models/model-epoch-{:02d}.ckpt'.format(epoch+1)))
            time_elapsed = time.time() - since
            print('=> Running time in a epoch: {:.0f}h {:.0f}m {:.0f}s'
                  .format(time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))
        scheduler.step()
        print()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        default='/home/mtb/ongoing_analysis/pku-autonomous-driving/attic/programmers_face_recognition/dataset',
                        help='input directory for images.')

    parser.add_argument('--model_name', type=str, default='transformer',
                        help='transformer, base.')

    parser.add_argument('--load_model', type=bool, default=False,
                        help='load_model.')

    parser.add_argument('--bn-mom', type=float, default=0.05,
                        help='bn-momentum (0.05)')
    
    parser.add_argument('--embed_size', type=int, default=128,
                        help='size of embedding. (512)')

    parser.add_argument('--num_classes', type=int, default=6,
                        help='the number of classes. (6)')

    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout. (0.1)')
    
    parser.add_argument('--metric-loss-coeff', type=float, default=0.2,
                        help='')
    
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for training. (0.001)')

    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping. (0.25)')

    parser.add_argument('--step_size', type=int, default=50,
                        help='period of learning rate decay. (5)')

    parser.add_argument('--gamma', type=float, default=0.5,
                        help='multiplicative factor of learning rate decay. (0.1)')

    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='the number of epochs. (100)')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size. (64) / (256)')

    parser.add_argument('--num_workers', type=int, default=16,
                        help='the number of processes working on cpu. (16)')

    parser.add_argument('--save_step', type=int, default=1,
                        help='save step of model. (1)')

    args = parser.parse_args()

    main(args)
