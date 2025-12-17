"""
 Python script to pre-compute cloud coverage statistics on the data of SEN12MS-CR-TS.
    The data loader performs online sampling of input and target patches depending on its flags
    (e.g.: split, region, n_input_samples, min_cov, max_cov, ) and the patches' calculated cloud coverage.
    If using sampler='random', patches can also vary across epochs to act as data augmentation mechanism.

    However, online computing of cloud masks can slow down data loading. A solution is to pre-compute
    cloud coverage an relief the dataloader from re-computing each sample, which is what this script offers. 
    Currently, pre-calculated statistics are exported in an *.npy file, a collection of which is readily
    available for download via https://syncandshare.lrz.de/getlink/fiHhwCqr7ch3X39XoGYaUGM8/splits
    
    Pre-computed statistics can be imported via the dataloader's "import_data_path" argument.
"""

import argparse
import os
import sys
import time
import random
import numpy as np
from tqdm import tqdm

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# see: https://docs.python.org/3/library/resource.html#resource.RLIM_INFINITY
resource.setrlimit(resource.RLIMIT_NOFILE, (int(1024*1e3), rlimit[1]))

import torch
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(dirname))
from data.dataLoader import SEN12MSCRTS

# fix all RNG seeds
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)


pathify = lambda path_list: [os.path.join(*path[0].split('/')[-6:]) for path in path_list]


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-compute cloud coverage statistics for SEN12MS-CR-TS.")
    parser.add_argument("--root", default="/home/data/SEN12MSCRTS", help="SEN12MS-CR-TS根目录")
    parser.add_argument("--split", default="test", choices=["all", "train", "val", "test"], help="数据划分")
    parser.add_argument("--region", default="all", choices=["all", "africa", "america", "asiaEast", "asiaWest", "europa"], help="数据区域")
    parser.add_argument("--sample-type", dest="sample_type", default="generic", choices=["generic", "cloudy_cloudfree"], help="采样类型")
    parser.add_argument("--input-t", dest="input_t", type=int, default=3, help="输入时间点数量")
    parser.add_argument("--import-data-path", dest="import_data_path", default=None, help="预加载索引文件路径")
    parser.add_argument("--export-data-path", dest="export_data_path", default=os.path.join(dirname, "precomputed"), help="导出预计算文件目录或文件路径，设为''关闭导出")
    parser.add_argument("--vary", default=None, choices=["random", "fixed"], help="是否在不同epoch随机重采样")
    parser.add_argument("--n-epochs", dest="n_epochs", type=int, default=None, help="预计算轮数，默认依据vary和sample_type推断")
    parser.add_argument("--max-samples", dest="max_samples", type=int, default=int(1e9), help="最多采样的patch数量")
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=0, help="DataLoader并行worker数量")
    parser.add_argument("--shuffle", action="store_true", help="是否打乱DataLoader顺序（导出索引时会被强制关闭）")
    parser.add_argument("--seed", type=int, default=1, help="随机种子")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # 更新随机种子
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    g.manual_seed(seed)

    split = args.split
    sample_type = args.sample_type
    input_t = args.input_t
    region = args.region
    import_data_path = args.import_data_path

    export_data_path = args.export_data_path if args.export_data_path else None
    if export_data_path:
        # 如果传入的是目录则直接创建；如果是文件则创建其父目录
        target_dir = export_data_path if not export_data_path.endswith('.npy') else os.path.dirname(export_data_path) or '.'
        os.makedirs(target_dir, exist_ok=True)
    vary = args.vary if args.vary is not None else ('random' if split != 'test' else 'fixed')
    n_epochs = args.n_epochs if args.n_epochs is not None else (1 if vary == 'fixed' or sample_type == 'generic' else 30)
    max_samples = args.max_samples

    shuffle = args.shuffle and export_data_path is None  # 导出索引时保持顺序

    sen12mscrts = SEN12MSCRTS(args.root, split=split, sample_type=sample_type, n_input_samples=input_t, region=region, sampler=vary, import_data_path=import_data_path)
    # instantiate dataloader, note: worker_init_fn is needed to get reproducible random samples across runs if vary_samples=True
    # note: if using 'export_data_path' then keep batch_size at 1 (unless moving data writing out of dataloader)
    #                                   and shuffle=False (processes patches in order, but later imports can still shuffle this)
    dataloader = torch.utils.data.DataLoader(sen12mscrts, batch_size=1, shuffle=shuffle, worker_init_fn=seed_worker, generator=g, num_workers=args.num_workers)
    
    if export_data_path is not None: 
        data_pairs  = {}  # collect pre-computed dates in a dict to be exported
        epoch_count = 0   # count, for loading time points that vary across epochs
    collect_var = []      # collect variance across S2 intensities

    # iterate over data to pre-compute indices for e.g. training or testing
    start_timer = time.time()
    for epoch in range(1, n_epochs + 1):
        print(f'\nCurating indices for {epoch}. epoch.')
        for pdx, patch in enumerate(tqdm(dataloader)):
            # stop sampling when sample count is exceeded
            if pdx>=max_samples: break
            
            if sample_type == 'generic':
                # collect variances in all samples' S2 intensities, finally compute grand average variance
                collect_var.append(torch.stack(patch['S2']).var())

                if export_data_path is not None:
                    if sample_type == 'cloudy_cloudfree':
                        # compute epoch-sensitive index, such that exported dates can differ across epochs 
                        adj_pdx = epoch_count*dataloader.dataset.__len__() + pdx
                        # performs repeated writing to file, only use this for processes dedicated for exporting
                        # and if so, only use a single thread of workers (--num_threads 1), this ain't thread-safe
                        data_pairs[adj_pdx] = {'input':     patch['input']['idx'], 'target': patch['target']['idx'],
                                               'coverage':  {'input': patch['input']['coverage'],
                                                             'output': patch['output']['coverage']},
                                               'paths':     {'input':  {'S1': pathify(patch['input']['S1 path']),
                                                                        'S2': pathify(patch['input']['S2 path'])},
                                                             'output': {'S1': pathify(patch['target']['S1 path']),
                                                                        'S2': pathify(patch['target']['S2 path'])}}}
                    elif sample_type == 'generic':
                        # performs repeated writing to file, only use this for processes dedicated for exporting
                        # and if so, only use a single thread of workers (--num_threads 1), this ain't thread-safe
                        data_pairs[pdx] = {'coverage':  patch['coverage'],
                                           'paths':     {'S1': pathify(patch['S1 path']),
                                                         'S2': pathify(patch['S2 path'])}}
        if sample_type == 'generic':    
            # export collected dates
            # eiter do this here after each epoch or after all epochs
            if export_data_path is not None:
                ds = dataloader.dataset
                if os.path.isdir(export_data_path):
                    export_here = os.path.join(export_data_path, f'{sample_type}_{input_t}_{split}_{region}_{ds.cloud_masks}.npy')
                else:
                    export_here = export_data_path
                np.save(export_here, data_pairs)
                print(f'\nEpoch {epoch_count+1}/{n_epochs}: Exported pre-computed dates to {export_here}')

                # bookkeeping at the end of epoch
                epoch_count += 1

    print(f'The grand average variance of S2 samples in the {split} split is: {torch.mean(torch.tensor(collect_var))}')

    if export_data_path is not None: print('Completed exporting data.')

    # benchmark speed of dataloader when (not) using 'import_data_path' flag
    elapsed = time.time() - start_timer
    print(f'Elapsed time is {elapsed}')