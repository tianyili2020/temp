import numpy as np
import torch
from torch import nn, optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('run on', DEVICE)

CONSIDER_RD_COST = True

CLASS_BIAS_RATE = 0.7 #0.7

IMAGE_SIZE = 64

MARGIN_LEN = 0

NUM_CLASSES = 6

SUB_CU_WIDTH_LIST  = [64, 32, 16, 8, 32, 16, 32, 16,  8,  8, 32, 16,  8,  4,  4,  4]
SUB_CU_HEIGHT_LIST = [64, 32, 16, 8, 16, 32,  8,  8, 32, 16,  4,  4,  4, 32, 16,  8]
#                  64*64   32*32         16*16          8*8       32*16        16*32        32*8       16*8         8*32       8*16
#CU_SIZE_CLASS_LIST = [[0,1], [0,1,2,3,4,5], [0,1,2,3,4,5], [0,2,3], [0,2,3,4,5], [0,2,3,4,5], [0,2,3,5], [0,2,3,5], [0,2,3,4], [0,2,3,4],
                   #[0,3,5], [0,3,5], [0,3], [0,2,4], [0,2,4], [0,2]]
#                   32*4     16*4     8*4     4*32     4*16     4*8

assert(len(SUB_CU_HEIGHT_LIST) == len(SUB_CU_WIDTH_LIST))
n_cu_type = len(SUB_CU_WIDTH_LIST)
print(n_cu_type)

cost_values_per_CTU = 0
for i_cu_type in range(n_cu_type):
    cost_values_per_CTU += (64 // SUB_CU_HEIGHT_LIST[i_cu_type]) * (64 // SUB_CU_WIDTH_LIST[i_cu_type])

COST_BYTES_PER_CTU = cost_values_per_CTU * 8
print(cost_values_per_CTU, COST_BYTES_PER_CTU)


SUB_CU_WIDTH = 32
SUB_CU_HEIGHT = 32

NUM_CHANNELS = 1

class SubCNN64(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(SubCNN64, self).__init__(**kwargs)
        self.sub_cu_height = 64
        self.sub_cu_width = 64
        self.sub_conv64 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=4, stride=4), nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=4), nn.PReLU()
        )
        self.sub_fc64 = nn.Sequential(
            nn.Linear(4*4*16, 16), nn.PReLU(),
            nn.Linear(16, NUM_CLASSES), nn.Softmax()
        )
        self.pad = nn.ReplicationPad2d(padding=MARGIN_LEN)
        self.sub_conv_pad = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1+MARGIN_LEN, stride=1, groups=32), nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=1+MARGIN_LEN, stride=1, groups=32), nn.PReLU()
        )

    def forward(self, x, qp):
        x[:, 0:16,:,:] *= (torch.reshape(qp, [-1, 1, 1, 1])/51.0 + 0.5)
        if MARGIN_LEN > 0:
            x = self.sub_conv_pad(x)
        x = self.sub_conv64(x)
        x = x.reshape(-1, 4*4*16)
        sub_qp = torch.reshape(qp, [-1, 1])/51.0 + 0.5
        assert(x.shape[0] % sub_qp.shape[0] == 0)
        sub_qp = sub_qp.repeat(1, x.shape[0] // sub_qp.shape[0])
        sub_qp = torch.reshape(sub_qp, [-1, 1])
        x[:, 0:128] *= sub_qp
        x = self.sub_fc64(x)
        return x


class SubCNN32(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(SubCNN32, self).__init__(**kwargs)
        self.sub_cu_height = 32
        self.sub_cu_width = 32
        self.sub_conv32 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=4), nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=4), nn.PReLU(),
            nn.Conv2d(64, 256, kernel_size=2, stride=2), nn.PReLU()
        )
        self.sub_fc32 = nn.Sequential(
            nn.Linear(256, 128), nn.PReLU(),
            nn.Linear(128, NUM_CLASSES), nn.Softmax()
        )
        self.pad = nn.ReplicationPad2d(padding=MARGIN_LEN)
        self.sub_conv_pad = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1+MARGIN_LEN, stride=1, groups=32), nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=1+MARGIN_LEN, stride=1, groups=32), nn.PReLU()
        )

    def forward(self, x, qp):
        x[:, 0:16,:,:] *= (torch.reshape(qp, [-1, 1, 1, 1])/51.0 + 0.5)
        if MARGIN_LEN > 0:
            x = self.sub_conv_pad(x)
        x = self.sub_conv32(x) # shape [batch_size, n_chan_out(=256), 1, 1]
        assert(x.shape[2]==1 and x.shape[3]==1)
        x = x.reshape(-1, 256) # shape [batch_size * 2 * 2, n_chan_out(=256)]
        sub_qp = torch.reshape(qp, [-1, 1])/51.0 + 0.5
        assert(x.shape[0] % sub_qp.shape[0] == 0)
        sub_qp = sub_qp.repeat(1, x.shape[0] // sub_qp.shape[0])
        sub_qp = torch.reshape(sub_qp, [-1, 1])
        x[:, 0:128] *= sub_qp
        x = self.sub_fc32(x) # shape [batch_size, 6]
        return x


class SubCNN16(nn.Module):
    def __init__(self, in_channels, sub_cu_width, sub_cu_height, **kwargs):
        super(SubCNN16, self).__init__(**kwargs)
        # (example): SUB_CU_WIDTH = 32   SUB_CU_HEIGHT = 16
        if sub_cu_width > sub_cu_height:
            k_width = sub_cu_width // sub_cu_height
            k_height = 1
        elif sub_cu_height > sub_cu_width:
            k_height = sub_cu_height // sub_cu_width
            k_width = 1
        else:
            k_width = 1
            k_height = 1
        self.sub_cu_width = sub_cu_width
        self.sub_cu_height = sub_cu_height
        # (example): k_width = 2   k_height = 1
        self.sub_conv16 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(4 * k_height, 4 * k_width), stride=(4 * k_height, 4 * k_width)), nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2), nn.PReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=2), nn.PReLU()
        )
        self.sub_fc16 = nn.Sequential(
            nn.Linear(128, 64), nn.PReLU(),
            nn.Linear(64, NUM_CLASSES), nn.Softmax(dim=1)
        )
        self.pad = nn.ReplicationPad2d(padding=MARGIN_LEN)
        self.sub_conv_pad = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1+MARGIN_LEN, stride=1, groups=32), nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=1+MARGIN_LEN, stride=1, groups=32), nn.PReLU()
        )

    def forward(self, x, qp):
        x[:, 0:16,:,:] *= (torch.reshape(qp, [-1, 1, 1, 1])/51.0 + 0.5)
        if MARGIN_LEN > 0:
            x = self.sub_conv_pad(x)
        x = self.sub_conv16(x) # (example): shape [batch_size, n_chan_out(=16), 1, 1]
        assert(x.shape[2]==1 and x.shape[3]==1)
        x = x.reshape(-1, 128) # (example): shape [batch_size, n_chan_out(=16)]
        sub_qp = torch.reshape(qp, [-1, 1])/51.0 + 0.5
        assert(x.shape[0] % sub_qp.shape[0] == 0)
        sub_qp = sub_qp.repeat(1, x.shape[0] // sub_qp.shape[0])
        sub_qp = torch.reshape(sub_qp, [-1, 1])
        x[:, 0:64] *= sub_qp
        x = self.sub_fc16(x) # (example): shape [batch_size, 6]
        return x


class SubCNN8(nn.Module):
    def __init__(self, in_channels, sub_cu_width, sub_cu_height, **kwargs):
        super(SubCNN8, self).__init__(**kwargs)
        # (example): SUB_CU_WIDTH = 16   SUB_CU_HEIGHT = 8
        if sub_cu_width > sub_cu_height:
            k_width = sub_cu_width // sub_cu_height
            k_height = 1
        elif sub_cu_height > sub_cu_width:
            k_height = sub_cu_height // sub_cu_width
            k_width = 1
        else:
            k_width = 1
            k_height = 1
        self.sub_cu_width = sub_cu_width
        self.sub_cu_height = sub_cu_height
        # (example): k_width = 2   k_height = 1
        self.sub_conv8 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(4 * k_height, 4 * k_width), stride=(4 * k_height, 4 * k_width)), nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2), nn.PReLU()
        )
        self.sub_fc8 = nn.Sequential(
            nn.Linear(64, 32), nn.PReLU(),
            nn.Linear(32, NUM_CLASSES), nn.Softmax(dim=1)
        )
        self.pad = nn.ReplicationPad2d(padding=MARGIN_LEN)
        self.sub_conv_pad = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1+MARGIN_LEN, stride=1, groups=32), nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=1+MARGIN_LEN, stride=1, groups=32), nn.PReLU()
        )

    def forward(self, x, qp):
        x[:, 0:16,:,:] *= (torch.reshape(qp, [-1, 1, 1, 1])/51.0 + 0.5)
        if MARGIN_LEN > 0:
            x = self.sub_conv_pad(x)
        x = self.sub_conv8(x) # (example): shape [batch_size, n_chan_out(=16), 1, 1]
        assert(x.shape[2]==1 and x.shape[3]==1)
        x = x.reshape(-1, 64) # (example): shape [batch_size, n_chan_out(=16)]
        sub_qp = torch.reshape(qp, [-1, 1])/51.0 + 0.5
        assert(x.shape[0] % sub_qp.shape[0] == 0)
        sub_qp = sub_qp.repeat(1, x.shape[0] // sub_qp.shape[0])
        sub_qp = torch.reshape(sub_qp, [-1, 1])
        x[:, 0:32] *= sub_qp
        x = self.sub_fc8(x) # (example): shape [batch_size, 6]
        return x


class SubCNN4(nn.Module):
    def __init__(self, in_channels, sub_cu_width, sub_cu_height, **kwargs):
        super(SubCNN4, self).__init__(**kwargs)
        # (example): SUB_CU_WIDTH = 8   SUB_CU_HEIGHT = 4
        if sub_cu_width > sub_cu_height:
            k_width = sub_cu_width // sub_cu_height
            k_height = 1
        elif sub_cu_height > sub_cu_width:
            k_height = sub_cu_height // sub_cu_width
            k_width = 1
        else:
            k_width = 1
            k_height = 1
        self.sub_cu_width = sub_cu_width
        self.sub_cu_height = sub_cu_height
        # (example): k_width = 2   k_height = 1
        self.sub_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(2 * k_height, 2 * k_width), stride=(2 * k_height, 2 * k_width)), nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2), nn.PReLU()
        )
        self.sub_fc4 = nn.Sequential(
            nn.Linear(64, 32), nn.PReLU(),
            nn.Linear(32, NUM_CLASSES), nn.Softmax(dim=1)
        )
        self.pad = nn.ReplicationPad2d(padding=MARGIN_LEN)
        self.sub_conv_pad = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1+MARGIN_LEN, stride=1, groups=32), nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=1+MARGIN_LEN, stride=1, groups=32), nn.PReLU()
        )

    def forward(self, x, qp):
        x[:, 0:16,:,:] *= (torch.reshape(qp, [-1, 1, 1, 1])/51.0 + 0.5)
        if MARGIN_LEN > 0:
            x = self.sub_conv_pad(x)
        x = self.sub_conv4(x) # (example): shape [batch_size, n_chan_out(=16), 1, 1]
        assert(x.shape[2]==1 and x.shape[3]==1)
        x = x.reshape(-1, 64) # (example): shape [batch_size, n_chan_out(=16)]
        #print(x.shape)
        sub_qp = torch.reshape(qp, [-1, 1])/51.0 + 0.5
        assert(x.shape[0] % sub_qp.shape[0] == 0)
        sub_qp = sub_qp.repeat(1, x.shape[0] // sub_qp.shape[0])
        sub_qp = torch.reshape(sub_qp, [-1, 1])
        x[:, 0:32] *= sub_qp
        x = self.sub_fc4(x) # (example): shape [batch_size, 6]
        return x


class MainNetFull(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(MainNetFull, self).__init__(**kwargs)
        self.plain_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.PReLU(),
        )
        self.conv1 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.PReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.PReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.PReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.PReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.PReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.PReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.PReLU())
        self.conv8 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.PReLU())
        self.conv9 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.PReLU())
        self.conv10 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.PReLU())
        self.conv11 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.PReLU())
        self.conv12 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.PReLU())

        self.pad = nn.ReplicationPad2d(padding=MARGIN_LEN)
        self.sub_conv_pad = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1+MARGIN_LEN, stride=1, groups=32), nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=1+MARGIN_LEN, stride=1, groups=32), nn.PReLU()
        )

    def resi_unit1(self, x):
        h_conv1 = self.conv1(x)
        h_conv2 = self.conv2(h_conv1)
        return x + h_conv2

    def resi_unit2(self, x):
        h_conv1 = self.conv3(x)
        h_conv2 = self.conv4(h_conv1)
        return x + h_conv2

    def resi_unit3(self, x):
        h_conv1 = self.conv5(x)
        h_conv2 = self.conv6(h_conv1)
        return x + h_conv2

    def resi_unit4(self, x):
        h_conv1 = self.conv7(x)
        h_conv2 = self.conv8(h_conv1)
        return x + h_conv2

    def resi_unit5(self, x):
        h_conv1 = self.conv9(x)
        h_conv2 = self.conv10(h_conv1)
        return x + h_conv2

    def resi_unit6(self, x):
        h_conv1 = self.conv11(x)
        h_conv2 = self.conv12(h_conv1)
        return x + h_conv2

    def forward(self, x):
        x /= 255.0
        x = x.reshape(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
        x = self.plain_conv(x)
        x = self.resi_unit1(x)
        f64 = self.resi_unit2(x)
        f32 = self.resi_unit3(f64)
        f16 = self.resi_unit4(f32)
        f8 = self.resi_unit5(f16)
        f4 = self.resi_unit6(f8)
        if MARGIN_LEN > 0:
            f64 = self.pad(f64)
            f32 = self.pad(f32)
            f16 = self.pad(f16)
            f8 = self.pad(f8)
            f4 = self.pad(f4)

        return f64, f32, f16, f8, f4


def get_loss(y_predict, y_truth, cost, class_ratios):

    y_truth = y_truth.reshape(-1, 1)
    num_samples = y_truth.shape[0]
    y_truth_valid_int = (y_truth >= 0).long()
    y_truth_valid = (y_truth >= 0).float()
    y_truth_one_hot = torch.zeros(num_samples, NUM_CLASSES).to(y_truth.device)
    y_truth_one_hot = y_truth_one_hot.scatter_(1, y_truth * y_truth_valid_int, 1)

    cost = cost.reshape(-1, NUM_CLASSES)
    #print('-----------------------------------')
    #print(cost)
    cost_mask_valid = (cost > 0.5).float() * (cost < 1000.5).float()
    cost_mask_invalid = 1 - cost_mask_valid
    #print(cost_mask_valid)
    cost_min = torch.min(cost * cost_mask_valid + 10000 * cost_mask_invalid, dim=1, keepdim=True)[0]
    cost_max = torch.max(cost * cost_mask_valid, dim=1, keepdim=True)[0]
    num_samples_valid = torch.sum(torch.max(cost_mask_valid, dim=1)[0])
    sample_mask_valid = torch.max(cost_mask_valid, dim=1, keepdim=True)[0]
    #print(num_samples_valid)
    #print(cost)
    #print(cost_max)
    #print(cost_min)
    #print(cost_mask_valid)
    #print(cost_mask_invalid)
    cost_rela_mag = (cost - cost_min) * cost_mask_valid / 20.0 + (cost_max - cost_min) * cost_mask_invalid / 20.0

    #print(cost_rela_mag)

    loss_sum = 0
    num_samples_sum = 0

    for i_class in range(NUM_CLASSES):
        loss_sum_one_class = torch.sum(-y_truth_one_hot[:, i_class] * torch.log(y_predict[:, i_class] + 1e-12) * y_truth_valid[:, 0])
        #loss_sum_one_class += torch.sum(-(1 - y_truth_one_hot[:, i_class]) * torch.log(1 - y_predict[:, i_class] + 1e-12) * y_truth_valid[:, 0])
        num_samples_temp = (class_ratios[i_class] * torch.sum(y_truth_valid).float() + 1e-12)
        loss_mean = loss_sum_one_class / num_samples_temp
        #print(loss_sum_one_class, num_samples_temp)
        loss_sum += loss_mean * (num_samples_temp ** CLASS_BIAS_RATE)
        num_samples_sum += num_samples_temp ** CLASS_BIAS_RATE
    loss_cross_entropy = loss_sum / num_samples_sum

    loss_MSE_wrong_class = (1 - y_truth_one_hot) * (y_predict * y_predict)
    #print(loss_MSE_wrong_class)
    #print(cost_rela_mag * sample_mask_valid)
    #print(torch.max(cost_rela_mag * sample_mask_valid))
    loss_RD_cost = torch.sum(loss_MSE_wrong_class * cost_rela_mag * sample_mask_valid) / num_samples_valid

    loss = 1.0 * loss_cross_entropy + 1.0 * loss_RD_cost
    #loss = loss_cross_entropy

    return loss, loss_cross_entropy, loss_RD_cost


def get_accuracy(y_predict, y_truth):
    y_truth = y_truth.reshape(-1, 1)
    num_samples = y_truth.shape[0]
    y_truth_valid_int = (y_truth >= 0).long()
    y_truth_valid = (y_truth >= 0).float()
    y_truth_one_hot = torch.zeros(num_samples, NUM_CLASSES).to(y_truth.device)
    y_truth_one_hot = y_truth_one_hot.scatter_(1, y_truth * y_truth_valid_int, 1)

    correct_prediction = y_truth_valid * torch.reshape(torch.argmax(y_truth_one_hot, dim=1) == torch.argmax(y_predict, dim=1).float(), [-1, 1])
    accuracy = torch.sum(y_truth_valid * correct_prediction) / (torch.sum(y_truth_valid)+1e-12)
    return accuracy


net = None
sub_cu_length = min(SUB_CU_HEIGHT, SUB_CU_WIDTH)


class Net(nn.Module):
    def __init__(self, sub_cu_width, sub_cu_height, **kwargs):
        super(Net, self).__init__(**kwargs)
        self.plain_conv = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 32, kernel_size=3, padding=1), nn.PReLU(),
        )
        self.conv1 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.PReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.PReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.PReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.PReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.PReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.PReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.PReLU())
        self.conv8 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.PReLU())
        self.conv9 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.PReLU())
        self.conv10 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.PReLU())
        self.conv11 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.PReLU())
        self.conv12 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.PReLU())
        self.sub_cu_width = sub_cu_width
        self.sub_cu_height = sub_cu_height

        self.pad = nn.ReplicationPad2d(padding=MARGIN_LEN)
        self.sub_conv_pad = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1+MARGIN_LEN, stride=1, groups=32), nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=1+MARGIN_LEN, stride=1, groups=32), nn.PReLU()
        )

        self.sub_conv64 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=4, stride=4), nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=4), nn.PReLU()
        )
        self.sub_fc64 = nn.Sequential(
            nn.Linear(4*4*16, 16), nn.PReLU(),
            nn.Linear(16, NUM_CLASSES), nn.Softmax(dim=1)
        )

        self.sub_conv32 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=4, stride=4), nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=4), nn.PReLU(),
            nn.Conv2d(64, 256, kernel_size=2, stride=2), nn.PReLU()
        )
        self.sub_fc32 = nn.Sequential(
            nn.Linear(256, 128), nn.PReLU(),
            nn.Linear(128, NUM_CLASSES), nn.Softmax(dim=1)
        )

        if sub_cu_width > sub_cu_height:
            k_width = sub_cu_width // sub_cu_height
            k_height = 1
        elif sub_cu_height > sub_cu_width:
            k_height = sub_cu_height // sub_cu_width
            k_width = 1
        else:
            k_width = 1
            k_height = 1

        # (example): k_width = 2   k_height = 1
        self.sub_conv16 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(4 * k_height, 4 * k_width), stride=(4 * k_height, 4 * k_width)), nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2), nn.PReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=2), nn.PReLU()
        )
        self.sub_fc16 = nn.Sequential(
            nn.Linear(128, 64), nn.PReLU(),
            nn.Linear(64, NUM_CLASSES), nn.Softmax(dim=1)
        )

        self.sub_conv8 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(4 * k_height, 4 * k_width), stride=(4 * k_height, 4 * k_width)), nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2), nn.PReLU()
        )
        self.sub_fc8 = nn.Sequential(
            nn.Linear(64, 32), nn.PReLU(),
            nn.Linear(32, NUM_CLASSES), nn.Softmax(dim=1)
        )

        self.sub_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(2 * k_height, 2 * k_width), stride=(2 * k_height, 2 * k_width)), nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2), nn.PReLU()
        )
        self.sub_fc4 = nn.Sequential(
            nn.Linear(64, 32), nn.PReLU(),
            nn.Linear(32, NUM_CLASSES), nn.Softmax(dim=1)
        )

    def resi_unit1(self, x):
        h_conv1 = self.conv1(x)
        h_conv2 = self.conv2(h_conv1)
        return x + h_conv2

    def resi_unit2(self, x):
        h_conv1 = self.conv3(x)
        h_conv2 = self.conv4(h_conv1)
        return x + h_conv2

    def resi_unit3(self, x):
        h_conv1 = self.conv5(x)
        h_conv2 = self.conv6(h_conv1)
        return x + h_conv2

    def resi_unit4(self, x):
        h_conv1 = self.conv7(x)
        h_conv2 = self.conv8(h_conv1)
        return x + h_conv2

    def resi_unit5(self, x):
        h_conv1 = self.conv9(x)
        h_conv2 = self.conv10(h_conv1)
        return x + h_conv2

    def resi_unit6(self, x):
        h_conv1 = self.conv11(x)
        h_conv2 = self.conv12(h_conv1)
        return x + h_conv2

    def forward(self, x, qp):
        x /= 255.0
        x = x.reshape(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
        x = self.plain_conv(x)
        x = self.resi_unit1(x)
        x = self.resi_unit2(x)
        sub_cu_length = min(self.sub_cu_height, self.sub_cu_width)
        if sub_cu_length <= 32:
            x = self.resi_unit3(x)
        if sub_cu_length <= 16:
            x = self.resi_unit4(x)
        if sub_cu_length <= 8:
            x = self.resi_unit5(x)
        if sub_cu_length == 4:
            x = self.resi_unit6(x)

        #print(x.shape)
        #print(qp.shape)
        #x = x * (torch.reshape(qp, [-1, 1, 1, 1])/51.0)
        #print(torch.reshape(qp, [-1, 1, 1, 1])/51.0 + 0.5)

        x[:, 0:16,:,:] *= (torch.reshape(qp, [-1, 1, 1, 1])/51.0 + 0.5)

        if MARGIN_LEN > 0:
            x_height = x.shape[2]
            x_width = x.shape[3]
            x_extend = self.pad(x)
            n_sample = x.shape[0]
            n_channel = x.shape[1]
            n_line = x_height // self.sub_cu_height
            n_col = x_width // self.sub_cu_width
            x_sub = torch.zeros([n_sample, n_line, n_col, n_channel, self.sub_cu_height + 2 * MARGIN_LEN, self.sub_cu_width + 2 * MARGIN_LEN]).to(x.device)
            for i_line in range(n_line):
                for i_col in range(n_col):
                    y_start = i_line * self.sub_cu_height
                    y_end = (i_line + 1) * self.sub_cu_height + 2 * MARGIN_LEN
                    x_start = i_col * self.sub_cu_width
                    x_end = (i_col + 1) * self.sub_cu_width + 2 * MARGIN_LEN
                    #print(x_sub.shape)
                    #print(y_start, y_end, x_start, x_end)
                    #print(x_extend[:, :, y_start : y_end, x_start : x_end].shape)
                    #print(x_sub[:, i_line, i_col, :, :, :].shape)
                    x_sub[:, i_line, i_col, :, :, :] = x_extend[:, :, y_start : y_end, x_start : x_end]
            x_sub = torch.reshape(x_sub, [-1, n_channel, self.sub_cu_height + 2 * MARGIN_LEN, self.sub_cu_width + 2 * MARGIN_LEN])
            x_sub = self.sub_conv_pad(x_sub)
            x = x_sub

        if sub_cu_length == 64:
            x = self.sub_conv64(x)
            x = x.reshape(-1, 4*4*16)
            sub_qp = torch.reshape(qp, [-1, 1])/51.0 + 0.5
            assert(x.shape[0] % sub_qp.shape[0] == 0)
            sub_qp = sub_qp.repeat(1, x.shape[0] // sub_qp.shape[0])
            sub_qp = torch.reshape(sub_qp, [-1, 1])
            x[:, 0:128] *= sub_qp
            x = self.sub_fc64(x)
        elif sub_cu_length == 32:
            x = self.sub_conv32(x) # shape [batch_size, n_chan_out(=256), 2, 2]
            x = x.transpose(1, 2)
            x = x.transpose(2, 3) # shape [batch_size, 2, 2, n_chan_out(=256)]
            x = x.reshape(-1, 256) # shape [batch_size * 2 * 2, n_chan_out(=256)]
            sub_qp = torch.reshape(qp, [-1, 1])/51.0 + 0.5
            assert(x.shape[0] % sub_qp.shape[0] == 0)
            sub_qp = sub_qp.repeat(1, x.shape[0] // sub_qp.shape[0])
            sub_qp = torch.reshape(sub_qp, [-1, 1])
            x[:, 0:128] *= sub_qp
            x = self.sub_fc32(x) # shape [batch_size * 2 * 2, 6]
        elif sub_cu_length == 16:
            x = self.sub_conv16(x) # (example): shape [batch_size, n_chan_out(=16), 4, 2]
            x = x.transpose(1, 2)
            x = x.transpose(2, 3) # (example): shape [batch_size, 4, 2, n_chan_out(=16)]
            #print(x.shape)
            x = x.reshape(-1, 128) # (example): shape [batch_size * 4 * 2, n_chan_out(=16)]
            #print(x.shape)
            sub_qp = torch.reshape(qp, [-1, 1])/51.0 + 0.5
            assert(x.shape[0] % sub_qp.shape[0] == 0)
            sub_qp = sub_qp.repeat(1, x.shape[0] // sub_qp.shape[0])
            sub_qp = torch.reshape(sub_qp, [-1, 1])
            x[:, 0:64] *= sub_qp
            x = self.sub_fc16(x) # (example): shape [batch_size * 4 * 2, 6]
        elif sub_cu_length == 8:
            x = self.sub_conv8(x) # (example): shape [batch_size, n_chan_out(=16), 8, 4]
            x = x.transpose(1, 2)
            x = x.transpose(2, 3) # (example): shape [batch_size, 8, 4, n_chan_out(=16)]
            #print(x.shape)
            x = x.reshape(-1, 64) # (example): shape [batch_size * 8 * 4, n_chan_out(=16)]
            #print(x.shape)
            sub_qp = torch.reshape(qp, [-1, 1])/51.0 + 0.5
            assert(x.shape[0] % sub_qp.shape[0] == 0)
            sub_qp = sub_qp.repeat(1, x.shape[0] // sub_qp.shape[0])
            sub_qp = torch.reshape(sub_qp, [-1, 1])
            x[:, 0:32] *= sub_qp
            x = self.sub_fc8(x) # (example): shape [batch_size * 8 * 4, 6]
        elif sub_cu_length == 4:
            x = self.sub_conv4(x) # (example): shape [batch_size, n_chan_out(=16), 16, 8]
            x = x.transpose(1, 2)
            x = x.transpose(2, 3) # (example): shape [batch_size, 16, 8, n_chan_out(=16)]
            #print(x.shape)
            x = x.reshape(-1, 64) # (example): shape [batch_size * 16 * 8, n_chan_out(=16)]
            #print(x.shape)
            #print(x[:, 0:32].shape)
            #print(sub_qp.shape)
            sub_qp = torch.reshape(qp, [-1, 1])/51.0 + 0.5
            assert(x.shape[0] % sub_qp.shape[0] == 0)
            sub_qp = sub_qp.repeat(1, x.shape[0] // sub_qp.shape[0])
            sub_qp = torch.reshape(sub_qp, [-1, 1])
            x[:, 0:32] *= sub_qp
            x = self.sub_fc4(x) # (example): shape [batch_size * 16 * 8, 6]
        return x
