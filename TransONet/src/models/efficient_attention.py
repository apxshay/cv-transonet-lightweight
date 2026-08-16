import torch
from torch import nn
from torch.nn import functional as f
import spconv.pytorch as spconv

class EfficientAttention(nn.Module):

    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        # print("in channels: {}, key_channels:{}, head_count: {}, value_channels:{}".format(
        #     in_channels,key_channels,head_count,value_channels
        # ))

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)
        # self.keys = spconv.SparseConv2d(in_channels, key_channels, 1)
        # self.queries = spconv.SparseConv2d(in_channels, key_channels, 1)
        # self.values = spconv.SparseConv2d(in_channels, value_channels, 1)
        #
        # self.reprojection = spconv.SparseConv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n,N,c = input_.size()
        #print("input_shape: {}".format(input_.shape))

        #N = h*w
        #H = W = int((N) ** 0.5)
        input_ = input_.reshape(n, c, N, 1)
        #print("input_shape: {}".format(input_.shape))
        keys = self.keys(input_)
        #print("keys_shape: {}".format(keys.shape))
        keys = keys.reshape((n, self.key_channels, N))
        queries = self.queries(input_).reshape(n, self.key_channels, N)
        values = self.values(input_).reshape((n, self.value_channels, N))
        # sparse_input = spconv.SparseConvTensor.from_dense(input_)
        # if sparse_input.features.shape[1] != self.in_channels:
        #     sparse_input.features = sparse_input.features.view(-1, self.in_channels)
        # keys = self.keys(sparse_input)
        # keys = keys.dense().reshape((n, self.key_channels, N))
        # queries = self.queries(sparse_input)
        # queries = queries.dense().reshape(n, self.key_channels, N)
        # values = self.values(sparse_input)
        # values = values.dense().reshape((n, self.value_channels, N))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = f.softmax(keys[
                            :,
                            i * head_key_channels: (i + 1) * head_key_channels,
                            :
                            ], dim=2)
            query = f.softmax(queries[
                              :,
                              i * head_key_channels: (i + 1) * head_key_channels,
                              :
                              ], dim=1)
            value = values[
                    :,
                    i * head_value_channels: (i + 1) * head_value_channels,
                    :
                    ]
            context = key @ value.transpose(1, 2)
            # attended_value = (
            #         context.transpose(1, 2) @ query
            # ).reshape(n, head_value_channels, h, w)

            attended_value = (
                    context.transpose(1, 2) @ query
            ).reshape(n, head_value_channels, N)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)#.transpose(1,2)
        #print("aggregated_values_shape: {}".format(aggregated_values.shape))
        #print("value_channels: {}".format(self.value_channels))
        #aggregated_values = spconv.SparseConvTensor.from_dense(aggregated_values)
        #old_features_shape = aggregated_values.features.shape
        # if aggregated_values.features.shape[1] != self.in_channels:
        #     # print(" sparse aggregated_values_shape: {}".format(aggregated_values.features.shape))
        #     # print("{},{},{}".format(n,N,c))
        #
        #     features = aggregated_values.features.view(n*N,c)
        #     #print("features: {}".format(features.shape))
        #     aggregated_values = aggregated_values.replace_feature(features)
        # print("in_chans: {}".format(self.in_channels))
        # print("head_count: {}".format(self.head_count))
        # print("key_channels: {}".format(self.key_channels))
        # print("value_channels: {}".format(self.value_channels))

        reprojected_value = self.reprojection(aggregated_values.unsqueeze(3))
        #reprojected_value = reprojected_value.replace_feature(reprojected_value.features.view(old_features_shape))#.transpose(1, 2))
        #reprojected_value = reprojected_value.dense()


        #print("reprojected_value_shape: {}".format(reprojected_value.shape))

        attention = reprojected_value + input_
        #print("attention: {}".format(attention.shape))
        return attention.squeeze(3).transpose(1, 2)
