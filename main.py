import torch
import utility
import data
import model as _model
import loss as _loss
from option import args
# from trainer import Trainer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
from importlib import import_module

if checkpoint.ok:
    print(args.data_test)
    # args.data_test = args.data_test.split('+')
    loader = data.Data(args)
    model = _model.Model(args, checkpoint)
    # checkpoint.write_log(model)
    print(model)
    # print(model, file=checkpoint.log_file)
    print('Total params: %.2f' % (sum(p.numel() for p in model.parameters())))
    print('Total params: %.2f' % (sum(p.numel() for p in model.parameters())), file=checkpoint.log_file)
    loss = _loss.Loss(args, checkpoint) if not args.test_only else None
    module = import_module('trainer.' + args.trainer.lower())
    t = module.make_trainer(args, loader, model, loss, checkpoint)
    # t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()

