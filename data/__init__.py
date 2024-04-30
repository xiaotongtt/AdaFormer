from importlib import import_module

from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader
from torch.utils.data import _utils
class Data:
    def __init__(self, args):
        kwargs = {}
        if not args.cpu:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = True
        else:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = False

        self.loader_train = None
        if not args.test_only:
            module_train = import_module('data.' + args.data_train.lower())
            trainset = getattr(module_train, args.data_train)(args)
            self.loader_train = DataLoader(
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers = args.n_threads, collate_fn = default_collate,
                pin_memory = False,
                drop_last = False,
                worker_init_fn = None
            )

        self.loader_test = []
        
        for d in args.data_test:
            if d in ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']:
                if not args.benchmark_noise:
                    module_test = import_module('data.benchmark')
                    testset = getattr(module_test, 'Benchmark')(args, name = d,  train=False)
                else:
                    module_test = import_module('data.benchmark_noise')
                    testset = getattr(module_test, 'BenchmarkNoise')(
                        args,
                        train=False
                    )

            elif d in ['Set5_Multi_Mild', 'Set14_Multi_Mild', 'B100_Multi_Mild', 'Urban100_Multi_Mild',
                       'Set5_Multi_Moderate', 'Set14_Multi_Moderate', 'B100_Multi_Moderate', 'Urban100_Multi_Moderate',
                       'Set5_Multi_Severe', 'Set14_Multi_Severe', 'B100_Multi_Severe', 'Urban100_Multi_Severe']:
                module_test = import_module('data.benchmark_multi')
                # print(d)
                dl = d.split('_')[-1]
                testset = getattr(module_test, 'Benchmark')(args, name = d, degrade_level = dl.lower(), train=False)

            elif d in ['test2k', 'test4k', 'test8k']:
                module_test = import_module('data.div8k_test')
                testset = getattr(module_test, 'Div8k_Test')(args, name=d, train=False)

            else:
                # module_test = import_module('data.' + args.data_test.lower())
                # testset = getattr(module_test, args.data_test)(args, train=False)
                print(args.data_test)
                module_test = import_module('data.' + args.data_test[0].lower())

                testset = getattr(module_test, args.data_test[0])(args, train=False)

            self.loader_test.append(DataLoader(
                testset,
                batch_size=1,
                shuffle=False,
                num_workers=args.n_threads, collate_fn=_utils.collate.default_collate,
                pin_memory=False,
                drop_last=False,
                worker_init_fn=None
            ))


