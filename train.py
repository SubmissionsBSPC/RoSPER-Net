import sys
import os
from loader import get_loader
from micro import TRAIN, VAL, TEST
from models.Model import Model
sys.path.append(os.getcwd())
from utils.loss_function import BceDiceLoss
from utils.tools import continue_train, get_logger, calculate_params_flops,set_seed
import torch
import argparse
from test import test_epoch
from Trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        default="PH2",
        help="input datasets name including ISIC2017, ISIC2018, PH2, Kvasir, or BUSI",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default="8",
        help="input batch_size",
    )
    parser.add_argument(
        "--imagesize",
        type=int, 
        default=256,
        help="input image resolution.",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="log",
        help="input log folder: ./log",
    )
    parser.add_argument(
        "--continues",
        type=int,
        default=0,
        help="1: continue to run; 0: don't continue to run",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default='checkpoints',
        help="the checkpoint path of last model: ./checkpoints",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="gpu_id:",
    )
    parser.add_argument(
        "--random",
        type=int,
        default=42,
        help="random configure:",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=300,
        help="end epoch",
    )
    return parser.parse_args()
    

def main():
    args=parse_args()
    checkpoint_path=os.path.join(os.getcwd(),args.checkpoint,args.datasets)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    logger = get_logger('train', os.path.join(os.getcwd(),args.log,args.datasets))
    torch.cuda.set_device(args.gpu)
    print("Current device ID:", torch.cuda.current_device())
    set_seed(args.random)
    torch.cuda.empty_cache()
    model=Model(out_channels=[16,32,48,64,80],scale_factor=[1,2,4,8,16])
    model = model.cuda()
    calculate_params_flops(model,size=args.imagesize,logger=logger)
    criterion=BceDiceLoss()
    #set optim
    optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr = 0.001,
            betas = (0.9,0.999),
            eps = 1e-8,
            weight_decay = 1e-2,
            amsgrad = False
        )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = 50,
            eta_min = 0.00001,
            last_epoch = -1
        )
    start_epoch=0
    if args.continues:
        model,start_epoch,min_loss,optimizer=continue_train(model,optimizer,checkpoint_path)
        lr=optimizer.state_dict()['param_groups'][0]['lr']
        print(f'start_epoch={start_epoch},min_loss={min_loss},lr={lr}')
    train_loader=get_loader(args.datasets,args.batchsize,args.imagesize,mode=TRAIN)
    val_loader=get_loader(args.datasets,args.batchsize,args.imagesize,mode=VAL)
    trainer=Trainer(train_loader=train_loader,
                    val_loader=val_loader,
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    start_epoch=start_epoch,
                    end_epoch=args.epoch,
                    logger=logger,
                    checkpoint_path=checkpoint_path,
                    save_cycles=20)
    trainer.run()
    


if __name__ == '__main__':
    main()
