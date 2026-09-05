import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers import ResnetBlockFC, FCPlanenet
from torch_scatter import scatter_mean, scatter_max, scatter_sum
from torch_scatter.composite import scatter_softmax
from src.common import coordinate2index, normalize_coordinate, normalize_3d_coordinate, positional_encoding, \
    normalize_dynamic_plane_coordinate, ChangeBasis


from ..models.mvit_model import MViT
from ..models.dyvit import VisionTransformerDiffPruning, VisionTransformerTeacher
from ..models.plane_transformer import HybridPlaneTransformer
from ..models.losses import DistillDiffPruningLoss_dynamic
from ..models.fast_quant import fast_quant
from ..models.generic_transformer import Transformer
import pdb
import time
from src.utils.others import SineLayer, print_profile_row
def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out





class DynamicLocalPoolPointnet(nn.Module):
    """PointNet-based encoder network with ResNet blocks
    for each local point on the ground plane. Learns n_channels dynamic planes 

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension 
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature 
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
        pos_encoding (bool): positional encoding  Defaults to False.
        n_channels (int): number of learning planes Defaults to 3.
        plane_net (str): type of plane-prediction network. Defaults to 'FCPlanenet'.
    """

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, scatter_type='max', unet=False, unet_kwargs=None,
                 plane_resolution=None,
                 grid_resolution=None, plane_type='xz', padding=0.1, n_blocks=5, pos_encoding=False, n_channels=3,
                 plane_net='FCPlanenet', use_patch_tokens=False,
                 point_grid_attention=False, grid_attention_dim=16,
                 correct_plane_fusion=False, transformer_patch_size=2,
                 teacher_patch_tokens=False, plane_processor='dyvit',
                 transformer_dim=128, transformer_depth=4,
                 transformer_heads=4):
        super().__init__()
        self.c_dim = c_dim
        self.num_channels = n_channels

        if pos_encoding == True:
            dim = 60

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, c_dim)
        self.point_grid_attention = point_grid_attention
        if point_grid_attention:
            self.grid_attention = nn.Sequential(
                nn.Linear(c_dim + 2, grid_attention_dim), nn.SiLU(),
                nn.Linear(grid_attention_dim, 1))
            nn.init.zeros_(self.grid_attention[-1].weight)
            nn.init.zeros_(self.grid_attention[-1].bias)
            self.grid_attention_gate = nn.Parameter(torch.tensor(0.))
        planenet_hidden_dim = hidden_dim
        self.fc_plane_net = FCPlanenet(n_dim=dim, hidden_dim=hidden_dim)

        # Create FC layers based on the number of planes
        self.plane_params = nn.ModuleList([
            nn.Linear(planenet_hidden_dim, 3) for i in range(n_channels)
        ])

        self.plane_params_hdim = nn.ModuleList([
            nn.Linear(3, hidden_dim) for i in range(n_channels)
        ])

        self.actvn = SineLayer(in_features=c_dim, out_features=c_dim, is_first=True)
        self.hidden_dim = hidden_dim


        self.reso_plane = plane_resolution
        self.correct_plane_fusion = correct_plane_fusion
        self.plane_processor = plane_processor
        if correct_plane_fusion and plane_processor != 'hybrid_transformer':
            self.plane_fusion = nn.Conv2d(n_channels * c_dim, c_dim, 1, bias=False)
            with torch.no_grad():
                weights = torch.eye(c_dim).repeat(1, n_channels) / n_channels
                self.plane_fusion.weight.copy_(weights[:, :, None, None])
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding
        self.optimizer = None

        if plane_processor == 'hybrid_transformer':
            self.transformer = HybridPlaneTransformer(
                in_channels=c_dim, num_planes=n_channels,
                dim=transformer_dim, depth=transformer_depth,
                num_heads=transformer_heads)
            print('plane processor: hybrid transformer')
        else:
            PRUNING_LOC = [1]
            self.KEEP_RATE = [0.7]
            self.transformer = VisionTransformerDiffPruning(
                img_size=self.reso_plane, patch_size=transformer_patch_size, embed_dim=c_dim, depth=2, in_chans=c_dim,
                num_classes=c_dim * self.reso_plane ** 2,
                num_heads=8, mlp_ratio=4, qkv_bias=True,
                pruning_loc=PRUNING_LOC, token_ratio=self.KEEP_RATE, distill=True, drop_path_rate=0.0,
                use_patch_tokens=use_patch_tokens
            ).cuda()
            self.teacher_model = VisionTransformerTeacher(
                img_size=self.reso_plane, patch_size=transformer_patch_size, embed_dim=c_dim, depth=2, in_chans=c_dim,
                num_classes=c_dim * self.reso_plane ** 2,
                num_heads=8, mlp_ratio=4, qkv_bias=True,
                use_patch_tokens=teacher_patch_tokens).cuda()
            self.criterion = DistillDiffPruningLoss_dynamic(
                self.teacher_model, torch.nn.CrossEntropyLoss(),
                clf_weight=0.0, keep_ratio=self.KEEP_RATE, mse_token=True,
                ratio_weight=2.0, distill_weight=0.5, dynamic=True)

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')

        self.pos_encoding = pos_encoding
        if pos_encoding:
            self.pe = positional_encoding()

    def project_dynamic_plane_features(self, p, c, normal_feature, basis_normalizer_matrix):
        # acquire indices of features in plane

        xy = normalize_dynamic_plane_coordinate(p.clone(), basis_normalizer_matrix,
                                                padding=self.padding)  # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane ** 2)

        point_features = c + normal_feature.unsqueeze(1)
        c = point_features.permute(0, 2, 1)
        fea_plane = scatter_mean(c, index, out=fea_plane)

        if self.point_grid_attention:
            cell_center = (torch.floor(xy * self.reso_plane) + 0.5) / self.reso_plane
            offset = (xy - cell_center) * self.reso_plane
            scores = self.grid_attention(
                torch.cat([point_features, offset], dim=2)).transpose(1, 2)
            weights = scatter_softmax(scores, index, dim=2)
            attended = scatter_sum(c * weights, index, out=torch.zeros_like(fea_plane))
            gate = torch.sigmoid(self.grid_attention_gate)
            fea_plane = fea_plane + gate * (attended - fea_plane)
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane,
                                      self.reso_plane)  # sparce matrix (B x 512 x reso x reso)
        return fea_plane

    def generate_dynamic_plane_features(self, p, c, normal_feature, basis_normalizer_matrix):
        if self.plane_processor == 'hybrid_transformer':
            planes = [self.project_dynamic_plane_features(
                p, c, normal_feature[l], basis_normalizer_matrix[:, l])
                for l in range(self.num_channels)]
            fea_plane = torch.stack(planes, dim=1)
        elif self.correct_plane_fusion:
            planes = [self.project_dynamic_plane_features(
                p, c, normal_feature[l], basis_normalizer_matrix[:, l])
                for l in range(self.num_channels)]
            fea_plane = self.plane_fusion(torch.cat(planes, dim=1))
        else:
            fea_plane = self.project_dynamic_plane_features(
                p, c, normal_feature, basis_normalizer_matrix)

        loss_dvit = None
        fea_plane_before = fea_plane

        # profiling: encoder (end)
        self.end_enc_prof(p)

        if self.plane_processor == 'hybrid_transformer':
            self.start_dyvit_prof(p)
            fea_plane = self.transformer(fea_plane)
            self.end_dyvit_prof(p)
            if self.training:
                return fea_plane, fea_plane.new_zeros(())
            return fea_plane

        if self.training:
            B, H, W, C = fea_plane.shape
            # profiling: dyvit (start)
            self.start_dyvit_prof(p)
            fea_plane, token_pred, mask, out_pred_score = self.transformer(fea_plane)
            # profiling: dyvit (end)
            self.end_dyvit_prof(p)

            outputs = [fea_plane, token_pred, mask, out_pred_score]

            loss_dvit, loss_dvit_part = self.criterion(fea_plane_before, outputs)
            fea_plane = fea_plane.reshape(B, H, W, C)


        else:
            B, H, W, C = fea_plane.shape
            # profiling: dyvit (start)
            self.start_dyvit_prof(p)
            fea_plane = self.transformer(fea_plane)
            # profiling: dyvit (end)
            self.end_dyvit_prof(p)
            fea_plane = fea_plane.reshape(B, H, W, C)

        if self.training:
            return fea_plane, loss_dvit
        else:
            return fea_plane

    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            if key == 'grid':
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_grid ** 3)
            else:
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_plane ** 2)

            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    def forward(self, p, optimizer):
        # print(p.size())
        batch_size, T, D = p.size()
        self.device = 'cpu'
        self.optimizer = optimizer
        # profiling: encoder (start)
        self.start_enc_prof(p)
        ##################
        if self.pos_encoding:
            pp = self.pe(p)
            net = self.fc_pos(pp)
            net_pl = self.fc_plane_net(pp)
        else:
            net = self.fc_pos(p)
            net_pl = self.fc_plane_net(p)
        ##################

        normal_fea = []
        normal_fea_hdim = {}

        for l in range(self.num_channels):
            normal_fea.append(self.plane_params[l](self.actvn(net_pl)))
            normal_fea_hdim['plane{}'.format(l)] = self.plane_params_hdim[l](normal_fea[l])

        self.plane_parameters = torch.stack(normal_fea, dim=1)  # plane parameter (batch_size x L x 3)
        C_mat = ChangeBasis(self.plane_parameters,
                            device=self.device)  # change of basis and normalizer matrix (concatenated)
        num_planes = C_mat.size()[1]

        # acquire the index for each point
        coord = {}
        index = {}

        for l in range(num_planes):
            coord['plane{}'.format(l)] = normalize_dynamic_plane_coordinate(p.clone(), C_mat[:, l],
                                                                            padding=self.padding)
            index['plane{}'.format(l)] = coordinate2index(coord['plane{}'.format(l)], self.reso_plane)

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        c = self.fc_c(net)

        fea = {}
        fea_loss = {}
        plane_loss = 0
        if self.correct_plane_fusion:
            normal_fea_hdims = [normal_fea_hdim['plane{}'.format(l)]
                                for l in range(C_mat.size(1))]
            C_mats = C_mat
        else:
            normal_fea_hdims = sum(normal_fea_hdim.values())
            C_mats = C_mat.sum(dim=1)
        if self.training:
            fea['planes'], fea_loss['planes_loss'] = self.generate_dynamic_plane_features(p, c, normal_fea_hdims, C_mats)
            plane_loss += fea_loss['planes_loss']
        else:
            fea['planes'] = self.generate_dynamic_plane_features(p, c, normal_fea_hdims, C_mats)


        fea['c_mat'] = C_mats


        # Normalize plane params for similarity loss calculation
        self.plane_parameters = self.plane_parameters.reshape([batch_size * num_planes, 3])
        self.plane_parameters = self.plane_parameters / torch.norm(self.plane_parameters, p=2, dim=1).view(
            batch_size * num_planes,
            1)  # normalize
        self.plane_parameters = self.plane_parameters.view(batch_size, -1)
        self.plane_parameters = self.plane_parameters.view(batch_size, -1, 3)


        if self.training:
            return fea, plane_loss
        else:
            return fea






    ### Profiling: encoder (start)
    def start_enc_prof(self, p):
        if getattr(self, 'enc_done', False):
            return
        self.enc_macs = 0
        self.enc_hooks = []

        def count_macs(mod, inp, out):
            if isinstance(mod, nn.Linear):
                self.enc_macs += out.numel() * mod.in_features

        skip = ('transformer', 'teacher_model')
        for name, mod in self.named_modules():
            if name.split('.')[0] in skip:
                continue
            if isinstance(mod, nn.Linear):
                self.enc_hooks.append(mod.register_forward_hook(count_macs))

        if p.is_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        self.t_enc = time.time()

    ### Profiling: encoder (end)
    def end_enc_prof(self, p):
        if getattr(self, 't_enc', None) is None:
            return
        if p.is_cuda:
            torch.cuda.synchronize()
        dt = time.time() - self.t_enc
        self.t_enc = None

        for h in self.enc_hooks:
            h.remove()
        self.enc_hooks = []

        if not hasattr(self, 'enc_times'):
            self.enc_times = []
        self.enc_times.append(dt)

        # 3 warmup + 10 timed
        if len(self.enc_times) < 13:
            return

        mean_t = sum(self.enc_times[3:]) / 10.0
        nparams = 0
        for name, par in self.named_parameters():
            if name.split('.')[0] in ('transformer', 'teacher_model'):
                continue
            nparams += par.numel()

        if p.is_cuda:
            mem_a = torch.cuda.max_memory_allocated() / 1e6
            mem_r = torch.cuda.max_memory_reserved() / 1e6
        else:
            mem_a, mem_r = 0.0, 0.0

        print_profile_row('Encoder', mean_t, mem_a, mem_r,
                          params=nparams, macs=self.enc_macs)
        self.enc_done = True

    ### Profiling: dyvit (start)
    def start_dyvit_prof(self, p):
        if getattr(self, 'dyvit_done', False):
            return
        self.dyvit_macs = 0
        self.dyvit_hooks = []

        def count_macs(mod, inp, out):
            if isinstance(mod, nn.Linear):
                self.dyvit_macs += out.numel() * mod.in_features
            elif isinstance(mod, nn.Conv2d):
                kernel = mod.kernel_size[0] * mod.kernel_size[1]
                self.dyvit_macs += out.numel() * mod.in_channels // mod.groups * kernel
            elif isinstance(mod, nn.ConvTranspose2d):
                kernel = mod.kernel_size[0] * mod.kernel_size[1]
                self.dyvit_macs += inp[0].numel() * mod.out_channels // mod.groups * kernel
            elif isinstance(mod, nn.MultiheadAttention):
                batch, tokens, channels = inp[0].shape
                self.dyvit_macs += batch * (4 * tokens * channels ** 2
                                             + 2 * tokens ** 2 * channels)

        mha_outputs = {id(mod.out_proj) for mod in self.transformer.modules()
                       if isinstance(mod, nn.MultiheadAttention)}
        for mod in self.transformer.modules():
            if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d,
                                nn.MultiheadAttention)) or (
                    isinstance(mod, nn.Linear) and id(mod) not in mha_outputs):
                self.dyvit_hooks.append(mod.register_forward_hook(count_macs))

        if p.is_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        self.t_dyvit = time.time()

    ### Profiling: dyvit (end)
    def end_dyvit_prof(self, p):
        if getattr(self, 't_dyvit', None) is None:
            return
        if p.is_cuda:
            torch.cuda.synchronize()
        dt = time.time() - self.t_dyvit
        self.t_dyvit = None

        for h in self.dyvit_hooks:
            h.remove()
        self.dyvit_hooks = []

        if not hasattr(self, 'dyvit_times'):
            self.dyvit_times = []
        self.dyvit_times.append(dt)

        # 3 warmup + 10 timed
        if len(self.dyvit_times) < 13:
            return

        mean_t = sum(self.dyvit_times[3:]) / 10.0
        nparams = 0
        for par in self.transformer.parameters():
            nparams += par.numel()

        if p.is_cuda:
            mem_a = torch.cuda.max_memory_allocated() / 1e6
            mem_r = torch.cuda.max_memory_reserved() / 1e6
        else:
            mem_a, mem_r = 0.0, 0.0

        block_name = ('Plane transformer' if self.plane_processor == 'hybrid_transformer'
                      else 'DyViT')
        print_profile_row(block_name, mean_t, mem_a, mem_r,
                          params=nparams, macs=self.dyvit_macs)
        self.dyvit_done = True




