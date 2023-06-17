from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer, data_prefetcher
import argparse
from torch.autograd import Variable
from trainer import UNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
try:
    from itertools import izip as zip
except ImportError:
    pass
import os
import sys
from torch.utils.tensorboard import SummaryWriter
import shutil
import random

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/unit_noise2clear-bn.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--trainer', type=str, default='UNIT', help="MUNIT|UNIT")
opts = parser.parse_args()

def main():
    cudnn.benchmark = True
    config = get_config(opts.config)
    max_iter = config['max_iter']
    display_size = config['display_size']
    config['vgg_model_path'] = opts.output_path
    device = config['device']

    trainer = UNIT_Trainer(config)

    if torch.cuda.is_available():
        trainer.cuda(config['gpuID'])

    train_loader_a, train_loader_b, train_loader_c, \
    test_loader_a, test_loader_b, test_loader_c = get_all_data_loaders(config)
    
    model_name = os.path.splitext(os.path.basename(opts.config))[0]
    writer = SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) 
    print('start training !!')
    
    iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
    
    TraindataA = data_prefetcher(train_loader_a) 
    TraindataB = data_prefetcher(train_loader_b)
    TraindataC = data_prefetcher(train_loader_c)
    testdataA = data_prefetcher(test_loader_a)
    testdataB = data_prefetcher(test_loader_b)
    testdataC = data_prefetcher(test_loader_c)

    while True:
        dataA = TraindataA.next()
        dataB = TraindataB.next()
        dataC = TraindataC.next()
        if dataA is None or dataB is None or dataC is None:
            TraindataA = data_prefetcher(train_loader_a) 
            TraindataB = data_prefetcher(train_loader_b)
            TraindataC = data_prefetcher(train_loader_c)

            dataA = TraindataA.next()
            dataB = TraindataB.next()
            dataC = TraindataC.next()
        with Timer("Elapsed time in update: %f"):

            for _ in range(3):
                trainer.content_update(dataA, dataB, dataC, config)
            
            trainer.dis_update(dataA, dataB, dataC, config)

            trainer.gen_update(dataA, dataB, dataC, config)

        trainer.update_learning_rate()
           
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, writer)
        if (iterations + 1) % config['image_save_iter'] == 0:
            testa = testdataA.next()
            testb = testdataB.next()
            testc = testdataC.next()
            if dataA is None or dataB is None or dataC is None \
                    or dataA.size(0) != display_size or dataB.size(0) != display_size or dataC.size(0) != display_size:
                testdataA = data_prefetcher(test_loader_a) 
                testdataB = data_prefetcher(test_loader_b)
                testdataC = data_prefetcher(test_loader_c)
                testa = testdataA.next()
                testb = testdataB.next()
                testc = testdataC.next()
            with torch.no_grad():
                test_image_outputs = trainer.sample(testa, testb, testc)
                train_image_outputs = trainer.sample(dataA, dataB, dataC)
            if test_image_outputs is not None and train_image_outputs is not None:
                write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
                write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
                write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

        if (iterations + 1) % config['image_display_iter'] == 0:
            with torch.no_grad():
                image_outputs = trainer.sample(dataA, dataB, dataC)
            if image_outputs is not None:
                write_2images(image_outputs, display_size, image_directory, 'train_current')

        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            writer.close()
            sys.exit('Finish training')
        

if __name__ == "__main__":
    main()    