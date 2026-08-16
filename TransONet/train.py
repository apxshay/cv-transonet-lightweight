import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import prune
import torch.nn as nn
import numpy as np
from src.utils.io import export_pointcloud
import os
import argparse
import time, datetime
import matplotlib; matplotlib.use('Agg')
from src import config, data
from src.checkpoints import CheckpointIO
from collections import defaultdict
import shutil
import pdb
from tqdm import tqdm, trange
from torchinfo import summary
from torchviz import make_dot

from src.models.fast_quant import fast_quant

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Arguments
parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds'
                         'with exit code 2.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")
# Set t0
t0 = time.time()

# Shorthands
out_dir = cfg['training']['out_dir']
batch_size = cfg['training']['batch_size']
backup_every = cfg['training']['backup_every']
vis_n_outputs = cfg['generation']['vis_n_outputs']
model_file = cfg['test']['model_file'] 
exit_after = args.exit_after

print(cfg['data']['pointcloud_noise'])


model_selection_metric = cfg['training']['model_selection_metric']
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')

# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

shutil.copyfile(args.config, os.path.join(out_dir, 'config.yaml'))

# Dataset
train_dataset = config.get_dataset('train', cfg)

val_dataset = config.get_dataset('val', cfg, return_idx=True)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=cfg['training']['n_workers'], shuffle=True,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, num_workers=cfg['training']['n_workers_val'], shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)


# For visualizations
vis_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)


model_counter = defaultdict(int)
data_vis_list = []
inputs=next(iter(train_loader))

# Model
model = config.get_model(cfg, device=device, dataset=train_dataset)

logger = SummaryWriter(os.path.join(out_dir, 'logs'))


# Build a data dictionary for visualization
iterator = iter(vis_loader)

for i in trange(len(vis_loader)):
    data_vis = next(iterator)
    idx = data_vis['idx'].item()
    model_dict = val_dataset.get_model_dict(idx)
    category_id = model_dict.get('category', 'n/a')
    category_name = val_dataset.metadata[category_id].get('name', 'n/a')
    category_name = category_name.split(',')[0]
    if category_name == 'n/a':
        category_name = category_id

    c_it = model_counter[category_id]
    if c_it < vis_n_outputs:
        data_vis_list.append({'category': category_name, 'it': c_it, 'data': data_vis})

    model_counter[category_id] += 1

# Model

# Intialize training
optimizer = optim.Adam(model.parameters(), lr=1e-4)
trainer = config.get_trainer(model, optimizer, cfg, device=device)

checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
print(f'out_dir {out_dir}')
try:
    print("train from file: "+str(model_file))
    load_dict = checkpoint_io.load(model_file)
except FileExistsError:
    print("new train")
    load_dict = dict()
epoch_it = load_dict.get('epoch_it', 0)
it = load_dict.get('it', 0)
metric_val_best = load_dict.get(
    'loss_val_best', -model_selection_sign * np.inf)

# Generator
generator = config.get_generator(model, cfg, optimizer, device=device)

if metric_val_best == np.inf or metric_val_best == -np.inf:
    metric_val_best = -model_selection_sign * np.inf

print('Current best validation metric (%s): %.8f'
      % (model_selection_metric, metric_val_best))



# Shorthands
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
validate_every = cfg['training']['validate_every']
visualize_every = cfg['training']['visualize_every']

# Print model
nparameters = sum(p.numel() for p in model.parameters())

print('Total number of parameters: %d' % nparameters)

print('output path: ', cfg['training']['out_dir'])


#while True:
for epoch in range(267): #epoch 13 scenes
    epoch_it += 1

    for batch in tqdm(train_loader):
        it += 1
        loss = trainer.train_step(cfg, batch)
        logger.add_scalar('train/loss', loss, it)
        
        if print_every > 0 and (it % print_every) == 0:
            t = datetime.datetime.now()
            print('[Epoch %02d] it=%03d, loss=%.4f, time: %.2fs, %02d:%02d'
                     % (epoch_it, it, loss, time.time() - t0, t.hour, t.minute))

        # Visualize output
        if visualize_every > 0 and (it % visualize_every) == 0:
        #if 200 > 0 and (it % 00) == 0:
            print('Visualizing')
            for data_vis in data_vis_list:
                datas = data_vis['data']
                out = generator.generate_mesh(datas)#data_vis['data'])
                pointcloud = generator.generate_pointcloud(datas)#data_vis['data'])
                #print(out[0])
                # Get statistics
                try:
                    mesh, stats_dict = out
                except TypeError:
                    mesh, stats_dict = out, {}
                pointcloud_path = os.path.join(out_dir, 'pointcloud')
                inputs_path = os.path.join(out_dir, 'input')
                occ_path = os.path.join(out_dir, 'occ')
                if not os.path.exists(pointcloud_path):
                    os.makedirs(pointcloud_path)
                if not os.path.exists(inputs_path):
                    print('make input folder:'+inputs_path)
                    os.makedirs(inputs_path)
                if not os.path.exists(occ_path):
                    print('make input folder:'+occ_path)
                    os.makedirs(occ_path)

                mesh.export(os.path.join(out_dir, 'vis', '{}_{}.obj'.format(data_vis['category'], data_vis['it'])))
                export_pointcloud(np.asarray(pointcloud.points), os.path.join(pointcloud_path, '{}_{}.ply'.format(data_vis['category'], data_vis['it'])))
                
                inputs_path = os.path.join(inputs_path, '{}_{}.ply'.format(data_vis['category'], data_vis['it']))
                inputs = datas['inputs'].squeeze(0).cpu().numpy()
                export_pointcloud(inputs, inputs_path, False)
                occ_path = os.path.join(occ_path, '{}_{}.ply'.format(data_vis['category'], data_vis['it']))
                occ = datas['points_iou'][datas['points_iou.occ'] == 1].squeeze(0).cpu().numpy()
                export_pointcloud(occ, occ_path, False)
                #print(datas['points_iou'][datas['points_iou.occ'] == 1])
                
        # Save checkpoint
        if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
            print('Saving checkpoint')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)

        # Backup if necessary
        if (backup_every > 0 and (it % backup_every) == 0):
            print('Backup checkpoint')
            checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
        # Run validation
        if validate_every > 0 and (it % validate_every) == 0:
        #if 100 > 0 and (it % 100) == 0:
            eval_dict = trainer.evaluate(val_loader)
            metric_val = eval_dict[model_selection_metric]
            print('Validation metric (%s): %.4f'
                  % (model_selection_metric, metric_val))

            for k, v in eval_dict.items():
                logger.add_scalar('val/%s' % k, v, it)

            if model_selection_sign * (metric_val - metric_val_best) > 0:
                metric_val_best = metric_val
                print('New best model (loss %.4f)' % metric_val_best)
                checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)

        # Exit if necessary
        if exit_after > 0 and (time.time() - t0) >= exit_after:
            print('Time limit reached. Exiting.')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
            exit(3)
