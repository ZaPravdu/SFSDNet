import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr


class Analyzer:
    def __init__(self, patch_layout=(4, 4), correlation='spearman'):
        self.patch_layout = patch_layout
        self.channel_keys = ['global', 'share', 'io']
        self.correlation = correlation
    def __call__(self, *args, **kwargs):
        pass

    def calculate_correlation(self, a_dict, b_dict):
        correlation_dict = {}
        for key in a_dict.keys():
            a_flat = a_dict[key].flatten()
            b_flat = b_dict[key].flatten()

            if self.correlation == 'pearson':
                correlation, p_value = pearsonr(a_flat, b_flat)
            elif self.correlation == 'spearman':
                correlation, p_value = spearmanr(a_flat, b_flat)
            else:
                raise Exception('correlation mode not supported')

            correlation_dict[key] = [float(correlation), float(p_value)]
        return correlation_dict

    def analyze_patch_error(self, outputs, targets):
        """
        output: list of pred [global, share, io] density map
        return: patch level error map
        """
        outputs_dict = self.pickup_data(outputs)
        targets_dict = self.pickup_data(targets)

        mae_maps = {}
        mse_maps = {}
        # calculate pixel error

        for key in self.channel_keys:
            mae, mse = self.calculate_patch_error(outputs_dict[key][0], targets_dict[key][0])

            mae_maps[key] = mae.squeeze().numpy() # [patches, patches]
            mse_maps[key] = mse.squeeze().numpy() # [patches, patches]

            # assert len(mae.shape()) == 3
        mae_maps['total'] = (mae_maps['global'] + mae_maps['share'] + mae_maps['io'])/3
        mse_maps['total'] = (mse_maps['global'] + mse_maps['share'] + mse_maps['io'])/3

        return mae_maps, mse_maps

    def calculate_patch_uncertainty(self, outputs):
        """
        outputs: list of forward result->forward result: list of pred global, share, io density map
        reture: dict of std: [global, share, io, total]
        """
        # 分项不确定性计算
        outputs_dict = self.pickup_data(outputs)

        # 计算标准差
        stds_dict = {}
        for key in self.channel_keys:
            output = outputs_dict[key]
            output = torch.concat(output, dim=1).detach().cpu()  # [augmented times, channel, H, W]
            assert output.shape[1] == 4

            output = self.patch_aggregation(output)  # [augmented times, 1, patches, patches]
            std = output.std(dim=1).squeeze()# [2, patches, patches]
            stds_dict[key] = std.numpy()

        stds_dict['total'] = (stds_dict['global'] + stds_dict['share'] + stds_dict['io'])/3

        return stds_dict

    def patch_aggregation(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.Tensor(data)

        aggregated_data = F.adaptive_avg_pool2d(data, output_size=self.patch_layout) #* self.patch_layout[0] * self.patch_layout[1]

        return aggregated_data

    def calculate_patch_error(self, output, target):
        mae = F.adaptive_avg_pool2d((target-output).abs(), output_size=self.patch_layout)
        mse = F.adaptive_avg_pool2d((target-output)**2, output_size=self.patch_layout)

        return mae, mse

    def pickup_data(self, data_list):
        """
        transform data from list to dict format
        data_list: list from model output
        return: dict of data
        """
        data_dict = {key: [] for key in self.channel_keys}

        # 分拣数据
        for data in data_list:
            for i, key in enumerate(self.channel_keys, start=0):
                data_dict[key].append(data[i].detach().cpu())

        return data_dict

