import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers import (
    ResnetBlockFC
)
from src.common import normalize_coordinate, normalize_3d_coordinate, coordinate2index, positional_encoding, normalize_dynamic_plane_coordinate
import pdb
import time
from src.utils.others import SineLayer, print_profile_row


class DynamicLocalDecoder(nn.Module):
    ''' Decoder for Dynamical Point Conv.
    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling mode  for points
        n_blocks (int): number of blocks ResNetBlockFC layers
        pos_encoding (bool): whether to use the positional encoing on input points
        padding (int): padding of input coordinates
        
    '''
    def __init__(self, dim=3, z_dim=128, c_dim=128,
                 hidden_size=256, leaky=False, sample_mode='bilinear',
                 n_blocks=5, pos_encoding=False, padding=0.1,
                 profile_macs=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.profile_macs = profile_macs
        
        if pos_encoding == True:
            dim = 60 # hardcoded

        if z_dim != 0:
            self.fc_z = nn.ModuleList([
                nn.Linear(z_dim, hidden_size) for i in range(n_blocks)
            ])
            self.conv_layers = nn.ModuleList([
                nn.ConvTranspose2d(2, 6, 3, stride=2, padding=1, output_padding=1,),
                nn.ConvTranspose2d(6, 12, 3, stride=2, padding=1, output_padding=1,),
                nn.ConvTranspose2d(12, 24, 3, stride=2, padding=1, output_padding=1,),
                nn.ConvTranspose2d(24, 48, 3, stride=2, padding=1, output_padding=1,),
                nn.ConvTranspose2d(48, 96, 3, stride=2, padding=1, output_padding=1,),
            ])
        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])


        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            # self.actvn = F.relu
            self.actvn_up = SineLayer(in_features=6, out_features=6)
            self.actvn_c = SineLayer(in_features=hidden_size, out_features=hidden_size)
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding

        self.pos_encoding = pos_encoding
        if pos_encoding:
            self.pe = positional_encoding()


    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def sample_dynamic_plane_feature(self, p, c, basis_normalizer_matrix):
        xy = normalize_dynamic_plane_coordinate(p.clone(), basis_normalizer_matrix, padding=self.padding) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def upsample_latent_code_to_feature_map(self, z_plane):
        batch_size = z_plane.shape[0]
        n_planes = 3

        z = z_plane.reshape(batch_size, 2, 4, 4)
        net = self.conv_layers[0](z)
        for conv_layer in self.conv_layers[1:]:
            # net = conv_layer(self.actvn(net))
            net = conv_layer(self.actvn_up(net))

        out_dict = {
            'xz': net[:, :32],
            'xy': net[:, 32:64],
            'yz': net[:, 64:],
        }
        return out_dict


    def forward(self, p, z_plane, c_plane, **kwargs):
        #print("p shape:{}".format(p.shape))
        # profiling: decoder (start)
        self.start_dec_prof(p)
        if self.z_dim > 0:
            # I do the reshaping
            z_plane = self.upsample_latent_code_to_feature_map(z_plane)

            num_planes = z_plane['c_mat'][1]

            z = 0
            for l in range(num_planes):
                z += self.sample_dynamic_plane_feature(p, z_plane['plane{}'.format(l)], z_plane['c_mat'][:,l])

            z = z.transpose(1, 2)

        if self.c_dim != 0:
            c = 0
            # print("just c_plane", type(c_plane))
            #print("c_plane", c_plane.keys())
            c_mat = c_plane['c_mat']
            planes = c_plane['planes']
            if c_mat.dim() == 4:
                for l in range(c_mat.size(1)):
                    c += self.sample_dynamic_plane_feature(p, planes[:, l], c_mat[:, l])
            else:
                c += self.sample_dynamic_plane_feature(p, planes, c_mat)
            c = c.transpose(1, 2)

        p = p.float()
        ##################
        if self.pos_encoding:
            p = self.pe(p)
        ##################
        
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.z_dim != 0:
                net = net + self.fc_z[i](z)
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        # out = self.fc_out(self.actvn(net))
        out = self.fc_out(self.actvn_c(net))
        out = out.squeeze(-1)

        # profiling: decoder (end)
        self.end_dec_prof(p)
        return out

    ### Profiling: decoder (start)
    def start_dec_prof(self, p):
        if not getattr(self, 'profile_macs', False):
            return
        if getattr(self, 'dec_done', False):
            return
        self.dec_macs = 0
        self.dec_hooks = []

        def count_macs(mod, inp, out):
            if isinstance(mod, nn.Linear):
                self.dec_macs += out.numel() * mod.in_features

        # thop misses scatter, grid_sample, MISE, marching cubes. expected.
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                self.dec_hooks.append(mod.register_forward_hook(count_macs))

        if p.is_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        self.t_dec = time.time()

    ### Profiling: decoder (end)
    def end_dec_prof(self, p):
        if not getattr(self, 'profile_macs', False):
            return
        if getattr(self, 't_dec', None) is None:
            return
        if p.is_cuda:
            torch.cuda.synchronize()
        dt = time.time() - self.t_dec
        self.t_dec = None

        for h in self.dec_hooks:
            h.remove()
        self.dec_hooks = []

        if not hasattr(self, 'dec_times'):
            self.dec_times = []
        self.dec_times.append(dt)

        # 3 warmup + 10 timed
        if len(self.dec_times) < 13:
            return

        mean_t = sum(self.dec_times[3:]) / 10.0
        nparams = 0
        for par in self.parameters():
            nparams += par.numel()

        if p.is_cuda:
            mem_a = torch.cuda.max_memory_allocated() / 1e6
            mem_r = torch.cuda.max_memory_reserved() / 1e6
        else:
            mem_a, mem_r = 0.0, 0.0

        print_profile_row('Decoder call', mean_t, mem_a, mem_r,
                          params=nparams, macs=self.dec_macs)
        self.dec_done = True

