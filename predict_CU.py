import numpy as np
import glob
import os
import sys
import time
from collections import Counter
import torch
import net_CTU64 as nt

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
torch.set_num_threads(1)

# file_yuv width  height  n_frame  INTRA_PERIOD  QP

UPPER_THRESHOLD = 1.0
MAX_INPUT_CTUS = 64

MARGIN_LEN = nt.MARGIN_LEN

QP = None

assert len(sys.argv) == 7 or len(sys.argv) == 1
if len(sys.argv) == 7:
    file_yuv = sys.argv[1]
    width = int(sys.argv[2])
    height = int(sys.argv[3])
    n_frame = int(sys.argv[4])
    intra_period = int(sys.argv[5])
    QP = int(sys.argv[6])
else:
    file_yuv = 'F:/YUV_All/BlowingBubbles_416x240_50.yuv'
    width = 416
    height = 240
    n_frame = 500
    intra_period = 8
    QP = 32

# when MARGIN_LEN == 4
#save_path = 'D:\Programs\Python\CU_Partition_VVC\Models_AI_qp22~37'

# when MARGIN_LEN == 0
#save_path = 'F:\WorkSpace_VVC\Encoder_evaluate_4Seqs_multi-qp_model_AI_Schedule\Models_AI_qp22~37_30000'
save_path = '/media/F/WorkSpace_VVC/Encoder_evaluate_4Seqs_multi-qp_model_AI_Schedule/Models_AI_qp22~37_200000'

#device = d2lt.try_gpu()
device = torch.device("cpu")

print('run on', device)

def print_current_line(str):
    sys.stdout.write('\r')
    sys.stdout.write(str)
    sys.stdout.flush()

def print_clear():
    sys.stdout.write('\r')
    sys.stdout.flush()

class FrameYUV(object):
    def __init__(self, Y, U, V):
        self._Y = Y
        self._U = U
        self._V = V

def read_YUV420_frame(fid, width, height, is_skip = False):

    if is_skip == False:
        d00 = height // 2
        d01 = width // 2
        Y_buf = fid.read(width * height)
        Y = np.reshape(np.frombuffer(Y_buf, dtype=np.uint8), [height, width])
        U_buf = fid.read(d01 * d00)
        U = np.reshape(np.frombuffer(U_buf, dtype=np.uint8), [d00, d01])
        V_buf = fid.read(d01 * d00)
        V = np.reshape(np.frombuffer(V_buf, dtype=np.uint8), [d00, d01])
        return FrameYUV(Y, U, V)
    else:
        fid.read(width * height * 3 // 2)
        return None

def print_count(count):
    for i in count.keys():
        print('%10d->%10d' % (i, count[i]))

def add_counts(count_list):
    count_sum = None
    for i in range(len(count_list)):
        if i == 0:
            count_sum = count_list[i]
        else:
            count_sum = Counter(dict(count_sum + count_list[i]))
    return count_sum

def read_one_csv_file(file):
    fid = open(file, 'rb')
    info = np.loadtxt(fid, delimiter=",", skiprows=0)
    fid.close()
    array = np.array(np.round(info).astype(np.uint32))
    array //= 10
    array[np.where(array % 10 == 0)] //= 10
    return array

def write_data_one_frame(fid_dst, frame_Y, label_array_list):
    height = frame_Y.shape[0]
    width = frame_Y.shape[1]
    n_line = height // 64
    n_col = width // 64
    for i_line in range(n_line):
        for i_col in range(n_col):
            y_start = i_line * 64
            x_start = i_col * 64
            frame_part = np.copy(frame_Y[y_start : y_start + 64, x_start : x_start + 64])
            fid_dst.write(frame_part.astype(np.uint8))
            for i_qp in range(len(label_array_list)):
                label_part = np.copy(label_array_list[i_qp][y_start // 4 : y_start // 4 + 16, x_start // 4 : x_start // 4 + 16])
                fid_dst.write(label_part.astype(np.uint16))


def mat_to_patches(mat, patch_len):

    mat_input = mat
    height = np.shape(mat)[0]
    width = np.shape(mat)[1]

    if height % patch_len != 0:
        margin_height = (height // patch_len + 1) * patch_len - height
        mat_input = np.concatenate([mat_input, np.zeros((margin_height, width))], axis=0)
        height += margin_height
    if width % patch_len != 0:
        margin_width = (width // patch_len + 1) * patch_len - width
        mat_input = np.concatenate([mat_input, np.zeros((height, margin_width))], axis=1)
        width += margin_width

    assert(height == mat_input.shape[0] and width == mat_input.shape[1])
    assert(width % patch_len == 0 and height % patch_len == 0)

    n_line = height // patch_len
    n_col = width // patch_len

    patches = np.zeros((n_line * n_col, patch_len, patch_len))
    info = np.zeros((n_line * n_col, 2)).astype(np.int32) # the second has 2 elements, representing x_start and y_start
    for i_line in range(n_line):
        for i_col in range(n_col):
            patches[i_line * n_col + i_col, :, :] = mat_input[i_line * patch_len : (i_line + 1) * patch_len, i_col * patch_len : (i_col + 1) * patch_len]
            info[i_line * n_col + i_col, :] = [i_col * patch_len, i_line * patch_len]

    return patches, info


class QuickList(object):
    def __init__(self):
        # basic settings
        self._initial_len = 100
        self._add_len = 100

        # variables
        self._len = self._initial_len
        self._num_element = 0
        self._list = [None] * self._initial_len

    def append(self, element):
        if self._num_element == self._len:
            self._list = self._list + [None] * self._add_len
            self._len += self._add_len
        self._list[self._num_element] = element
        self._num_element += 1

    def get_list(self):
        return self._list[0 : self._num_element]


def load_CU_info(i_frame, iCTU_start, CTU_info, CU_info_unfinished):
    for i in range(np.shape(CTU_info)[0]):
        CU_info_unfinished.append([i_frame, iCTU_start + i] + CTU_info[i, :].tolist() + [64, 64, None, None, None, None, None, None])


def update_CU_info(f64, f32, f16, f8, f4, qp, CU_info_remain, CU_info_finish):

    total_time = 0
    net_time = 0

    CU_info_remain_new = QuickList()

    CU_info_remain_s64 = QuickList()
    CU_info_remain_s32 = QuickList()
    CU_info_remain_s16 = QuickList()
    CU_info_remain_s8 = QuickList()
    CU_info_remain_32_16 = QuickList()
    CU_info_remain_16_32 = QuickList()
    CU_info_remain_32_8 = QuickList()
    CU_info_remain_16_8 = QuickList()
    CU_info_remain_8_32 = QuickList()
    CU_info_remain_8_16 = QuickList()
    CU_info_remain_32_4 = QuickList()
    CU_info_remain_16_4 = QuickList()
    CU_info_remain_8_4 = QuickList()
    CU_info_remain_4_32 = QuickList()
    CU_info_remain_4_16 = QuickList()
    CU_info_remain_4_8 = QuickList()

    for CU_info in CU_info_remain:
        i_frame = CU_info[0]
        i_CTU = CU_info[1]
        x_start = CU_info[2]
        y_start = CU_info[3]
        CU_width = CU_info[4]
        CU_height = CU_info[5]
        if (CU_width, CU_height) == (64, 64):
            CU_info_remain_s64.append(CU_info)
        elif (CU_width, CU_height) == (32, 32):
            CU_info_remain_s32.append(CU_info)
        elif (CU_width, CU_height) == (16, 16):
            CU_info_remain_s16.append(CU_info)
        elif (CU_width, CU_height) == (8, 8):
            CU_info_remain_s8.append(CU_info)
        elif (CU_width, CU_height) == (32, 16):
            CU_info_remain_32_16.append(CU_info)
        elif (CU_width, CU_height) == (16, 32):
            CU_info_remain_16_32.append(CU_info)
        elif (CU_width, CU_height) == (32, 8):
            CU_info_remain_32_8.append(CU_info)
        elif (CU_width, CU_height) == (16, 8):
            CU_info_remain_16_8.append(CU_info)
        elif (CU_width, CU_height) == (8, 32):
            CU_info_remain_8_32.append(CU_info)
        elif (CU_width, CU_height) == (8, 16):
            CU_info_remain_8_16.append(CU_info)
        elif (CU_width, CU_height) == (32, 4):
            CU_info_remain_32_4.append(CU_info)
        elif (CU_width, CU_height) == (16, 4):
            CU_info_remain_16_4.append(CU_info)
        elif (CU_width, CU_height) == (8, 4):
            CU_info_remain_8_4.append(CU_info)
        elif (CU_width, CU_height) == (4, 32):
            CU_info_remain_4_32.append(CU_info)
        elif (CU_width, CU_height) == (4, 16):
            CU_info_remain_4_16.append(CU_info)
        elif (CU_width, CU_height) == (4, 8):
            CU_info_remain_4_8.append(CU_info)

    def update(CU_info_onesize0, CU_width, CU_height, feature_maps, net):
        CU_info_onesize = CU_info_onesize0.get_list()
        #time_0 = time.time()
        n_CU = len(CU_info_onesize)
        if n_CU > 0:
            CU_patches = np.zeros((n_CU, 32, CU_height + 2 * MARGIN_LEN, CU_width + 2 * MARGIN_LEN)) # 32 is the number of feature maps in MainNet
            #CU_patches = torch.zeros((len(CU_info_onesize), 32, CU_height, CU_width))
            #print('t1 = %.6f' % (time.time() - time_0))

            for i_CU in range(n_CU):
                CU_info = CU_info_onesize[i_CU]
                #i_frame = CU_info[0]
                i_CTU = CU_info[1]
                x_start = CU_info[2]
                y_start = CU_info[3]
                #print(i_frame, i_CTU, x_start, y_start)
                assert(CU_width == CU_info[4])
                assert(CU_height == CU_info[5])
                CU_patches[i_CU, :, :, :] = feature_maps[i_CTU, :, y_start % 64 : y_start % 64 + CU_height + 2 * MARGIN_LEN, x_start % 64 : x_start % 64 + CU_width + 2 * MARGIN_LEN]

            #print('t2 = %.6f' % (time.time() - time_0))
            x_temp = torch.Tensor(CU_patches).to(device)
            #print('t3 = %.6f' % (time.time() - time_0))
            #time_1 = time.time()
            n_sample = x_temp.shape[0]
            qp_temp = qp.reshape([-1, 1]).repeat(n_sample, 1)
            class_probs = net(x_temp, qp_temp)
            #time_2 = time.time()
            #net_time = (time_2 - time_1)
            #print('t4 = %.6f' % (time.time() - time_0))

            class_probs = class_probs.cpu().data.numpy()
            #print('CU size = [%d  %d]' % (CU_width, CU_height))
            #print(class_probs)

            class_mask = np.array([True] * 6)
            if CU_width == 64 or CU_height == 64:
                class_mask[2:6] = False
            if CU_width != CU_height:
                class_mask[1] = False
            if CU_height < 8:
                class_mask[2] = False
            if CU_width < 8:
                class_mask[3] = False
            if CU_height < 16:
                class_mask[4] = False
            if CU_width < 16:
                class_mask[5] = False

            class_probs[:, np.where(class_mask == False)] = 0
            class_probs /= (np.sum(class_probs, axis=1, keepdims=True) + 1e-12)
            #n_class_valid = np.shape(np.where(class_mask == True)[0])[0]
            #print('t5 = %.6f' % (time.time() - time_0))

            for i_CU in range(n_CU):
                CU_info = CU_info_onesize[i_CU]
                i_frame = CU_info[0]
                i_CTU = CU_info[1]
                x_start = CU_info[2]
                y_start = CU_info[3]
                class_bool_list = (class_probs[i_CU, :] >= np.max(class_probs[i_CU, :]) * UPPER_THRESHOLD)
                class_probs_arg = np.argsort(class_probs[i_CU, :])
                # "class_bool_list_extend" is a 6-dim array. Its values represent the relative magnitudes in array "class_probs[i_CU, :]".
                # Specially, if an index satisfies class_bool_list[index]==True, add 16 to class_bool_list_extend[index].
                # As such, when VTM encoder finds that class_bool_list_extend[index]>=16, it will know this mode is selected.
                class_bool_list_extend = class_bool_list.astype(np.int16) * 16
                rela_mag = 0
                for i_temp in range(6):
                    if i_temp >= 1:
                        if class_probs_arg[i_temp] > class_probs_arg[i_temp - 1]:
                            rela_mag += 1
                    class_bool_list_extend[class_probs_arg[i_temp]] += rela_mag
                #print(class_probs[i_CU, :])
                #print(class_bool_list_extend)
                write_arr = [i_frame, i_CTU, x_start, y_start, CU_width, CU_height] + ((class_probs[i_CU, :] * 1022).astype(np.int16)).tolist()
                #CU_info_finish.append([i_frame, i_CTU, x_start, y_start, CU_width, CU_height] + class_bool_list_extend.tolist())
                #print(write_arr)
                CU_info_finish.append(write_arr)
                if class_bool_list[1] == True: # quad-tree
                    CU_info_remain_new.append([i_frame, i_CTU, x_start, y_start, CU_width // 2, CU_height // 2] + [-1] * 6)
                    CU_info_remain_new.append([i_frame, i_CTU, x_start + CU_width // 2, y_start, CU_width // 2, CU_height // 2] + [-1] * 6)
                    CU_info_remain_new.append([i_frame, i_CTU, x_start, y_start + CU_height // 2, CU_width // 2, CU_height // 2] + [-1] * 6)
                    CU_info_remain_new.append([i_frame, i_CTU, x_start + CU_width // 2, y_start + CU_height // 2, CU_width // 2, CU_height // 2] + [-1] * 6)
                if class_bool_list[2] == True: # binary-tree (horizontal)
                    CU_info_remain_new.append([i_frame, i_CTU, x_start, y_start, CU_width, CU_height // 2] + [-1] * 6)
                    CU_info_remain_new.append([i_frame, i_CTU, x_start, y_start + CU_height // 2, CU_width, CU_height // 2] + [-1] * 6)
                if class_bool_list[3] == True: # binary-tree (vertical)
                    CU_info_remain_new.append([i_frame, i_CTU, x_start, y_start, CU_width // 2, CU_height] + [-1] * 6)
                    CU_info_remain_new.append([i_frame, i_CTU, x_start + CU_width // 2, y_start, CU_width // 2, CU_height] + [-1] * 6)
                if class_bool_list[4] == True: # trinary-tree (horizontal)
                    CU_info_remain_new.append([i_frame, i_CTU, x_start, y_start, CU_width, CU_height // 4] + [-1] * 6)
                    CU_info_remain_new.append([i_frame, i_CTU, x_start, y_start + CU_height // 4, CU_width, CU_height // 2] + [-1] * 6)
                    CU_info_remain_new.append([i_frame, i_CTU, x_start, y_start + CU_height // 4 * 3, CU_width, CU_height // 4] + [-1] * 6)
                if class_bool_list[5] == True: # trinary-tree (vertical)
                    CU_info_remain_new.append([i_frame, i_CTU, x_start, y_start, CU_width // 4, CU_height] + [-1] * 6)
                    CU_info_remain_new.append([i_frame, i_CTU, x_start + CU_width // 4, y_start, CU_width // 2, CU_height] + [-1] * 6)
                    CU_info_remain_new.append([i_frame, i_CTU, x_start + CU_width // 4 * 3, y_start, CU_width // 4, CU_height] + [-1] * 6)
            #total_time = time.time() - time_0
            #print('Total time = %.6f.  Net time = %.6f.  Ratio = %.2f' % (total_time, net_time, total_time / net_time))
        return


    update(CU_info_remain_s64, 64, 64, f64, sub_net_s64)
    update(CU_info_remain_s32, 32, 32, f32, sub_net_s32)
    update(CU_info_remain_s16, 16, 16, f16, sub_net_s16)
    update(CU_info_remain_s8, 8, 8, f8, sub_net_s8)

    update(CU_info_remain_32_16, 32, 16, f16, sub_net_32_16)
    update(CU_info_remain_16_32, 16, 32, f16, sub_net_16_32)

    update(CU_info_remain_32_8, 32, 8, f8, sub_net_32_8)
    update(CU_info_remain_16_8, 16, 8, f8, sub_net_16_8)
    update(CU_info_remain_8_32, 8, 32, f8, sub_net_8_32)
    update(CU_info_remain_8_16, 8, 16, f8, sub_net_8_16)

    update(CU_info_remain_32_4, 32, 4, f4, sub_net_32_4)
    update(CU_info_remain_16_4, 16, 4, f4, sub_net_16_4)
    update(CU_info_remain_8_4,  8,  4, f4, sub_net_8_4)
    update(CU_info_remain_4_32, 4, 32, f4, sub_net_4_32)
    update(CU_info_remain_4_16, 4, 16, f4, sub_net_4_16)
    update(CU_info_remain_4_8,  4,  8, f4, sub_net_4_8)

    CU_info_remain.clear()
    CU_info_remain += CU_info_remain_new.get_list()


def get_latest_model(save_path, sub_cu_width, sub_cu_height, qp, type):
    if type == 'params':
        model_file_list = glob.glob(os.path.join(save_path, 'model_*_CU%dx%d.params') % (sub_cu_width, sub_cu_height))
    elif type == 'dat':
        model_file_list = glob.glob(os.path.join(save_path, 'loss_accuracy_list_*_CU%dx%d.dat') % (sub_cu_width, sub_cu_height))
    #print('--------------\nAvailable models:')
    n_model = len(model_file_list)
    assert(n_model >= 1)
    time_num_list = [None] * n_model
    for i_model in range(n_model):
        path, name = os.path.split(model_file_list[i_model])
        #print('  [%3d] %s' % (i_model + 1, name))
        str_list = str.split(name, '_')
        if type == 'params':
            time_num_list[i_model] = int(str_list[1]) * 1000000 + int(str_list[2])
        elif type == 'dat':
            time_num_list[i_model] = int(str_list[3]) * 1000000 + int(str_list[4])
    max_index = time_num_list.index(max(time_num_list))
    model_file = model_file_list[max_index]
    #print('load model: %s' % model_file)
    return model_file


def remove_state_dict_prefix(state_dict, str_prefix):
    len_str = len(str_prefix)
    key_list = []
    for (key, value) in state_dict.items():
        if key[0:len_str] == str_prefix:
            key_list.append(key)
    for key in key_list:
        state_dict[key[len_str:]] = state_dict.pop(key)


def safe_load_state_dict(net, state_dict):
    state_dict_new = net.state_dict()
    for (key, value) in state_dict.items():
        if key in state_dict_new:
            state_dict_new[key] = value
            #print('LOAD: key \'%s\'' % key)
        else:
            pass
            #print('WARNING: key \'%s\' is not in \'state_dict_new\'' % key)
    net.load_state_dict(state_dict_new)


# all possible functions for main network and sub-networks
main_net = nt.MainNetFull(in_channels=1) # input: 64x64 CTU, output: 5 groups of feature maps

# sub-networks for square CUs
sub_net_s64 = nt.SubCNN64(in_channels=32)
sub_net_s32 = nt.SubCNN32(in_channels=32)
sub_net_s16 = nt.SubCNN16(in_channels=32, sub_cu_width=16, sub_cu_height=16)
sub_net_s8 = nt.SubCNN8(in_channels=32, sub_cu_width=8, sub_cu_height=8)

# sub-networks for rectangular CUs
sub_net_32_16 = nt.SubCNN16(in_channels=32, sub_cu_width=32, sub_cu_height=16)
sub_net_16_32 = nt.SubCNN16(in_channels=32, sub_cu_width=16, sub_cu_height=32)
sub_net_32_8 = nt.SubCNN8(in_channels=32, sub_cu_width=32, sub_cu_height=8)
sub_net_16_8 = nt.SubCNN8(in_channels=32, sub_cu_width=16, sub_cu_height=8)
sub_net_8_32 = nt.SubCNN8(in_channels=32, sub_cu_width=8, sub_cu_height=32)
sub_net_8_16 = nt.SubCNN8(in_channels=32, sub_cu_width=8, sub_cu_height=16)
sub_net_32_4 = nt.SubCNN4(in_channels=32, sub_cu_width=32, sub_cu_height=4)
sub_net_16_4 = nt.SubCNN4(in_channels=32, sub_cu_width=16, sub_cu_height=4)
sub_net_8_4 = nt.SubCNN4(in_channels=32, sub_cu_width=8, sub_cu_height=4)
sub_net_4_32 = nt.SubCNN4(in_channels=32, sub_cu_width=4, sub_cu_height=32)
sub_net_4_16 = nt.SubCNN4(in_channels=32, sub_cu_width=4, sub_cu_height=16)
sub_net_4_8 = nt.SubCNN4(in_channels=32, sub_cu_width=4, sub_cu_height=8)

main_net.to(device)
sub_net_s64.to(device)
sub_net_s32.to(device)
sub_net_s16.to(device)
sub_net_s8.to(device)
sub_net_32_16.to(device)
sub_net_16_32.to(device)
sub_net_32_8.to(device)
sub_net_16_8.to(device)
sub_net_8_32.to(device)
sub_net_8_16.to(device)
sub_net_32_4.to(device)
sub_net_16_4.to(device)
sub_net_8_4.to(device)
sub_net_4_32.to(device)
sub_net_4_16.to(device)
sub_net_4_8.to(device)

# main net
state_dict = torch.load(get_latest_model(save_path, 32, 4, QP, 'params'))
remove_state_dict_prefix(state_dict, '0.')
safe_load_state_dict(main_net, state_dict)

# 64*64 CUs
state_dict = torch.load(get_latest_model(save_path, 64, 64, QP, 'params'))
remove_state_dict_prefix(state_dict, '1.')
safe_load_state_dict(sub_net_s64, state_dict)

# 32*32 CUs
state_dict = torch.load(get_latest_model(save_path, 32, 32, QP, 'params'))
remove_state_dict_prefix(state_dict, '1.')
safe_load_state_dict(sub_net_s32, state_dict)

# 16*16 CUs
state_dict = torch.load(get_latest_model(save_path, 16, 16, QP, 'params'))
remove_state_dict_prefix(state_dict, '1.')
safe_load_state_dict(sub_net_s16, state_dict)

# 8*8 CUs
state_dict = torch.load(get_latest_model(save_path, 8, 8, QP, 'params'))
remove_state_dict_prefix(state_dict, '1.')
safe_load_state_dict(sub_net_s8, state_dict)

# 32*16 CUs
state_dict = torch.load(get_latest_model(save_path, 32, 16, QP, 'params'))
remove_state_dict_prefix(state_dict, '1.')
safe_load_state_dict(sub_net_32_16, state_dict)

# 16*32 CUs
state_dict = torch.load(get_latest_model(save_path, 16, 32, QP, 'params'))
remove_state_dict_prefix(state_dict, '1.')
safe_load_state_dict(sub_net_16_32, state_dict)

# 32*8 CUs
state_dict = torch.load(get_latest_model(save_path, 32, 8, QP, 'params'))
remove_state_dict_prefix(state_dict, '1.')
safe_load_state_dict(sub_net_32_8, state_dict)

# 16*8 CUs
state_dict = torch.load(get_latest_model(save_path, 16, 8, QP, 'params'))
remove_state_dict_prefix(state_dict, '1.')
safe_load_state_dict(sub_net_16_8, state_dict)

# 8*32 CUs
state_dict = torch.load(get_latest_model(save_path, 8, 32, QP, 'params'))
remove_state_dict_prefix(state_dict, '1.')
safe_load_state_dict(sub_net_8_32, state_dict)

# 8*16 CUs
state_dict = torch.load(get_latest_model(save_path, 8, 16, QP, 'params'))
remove_state_dict_prefix(state_dict, '1.')
safe_load_state_dict(sub_net_8_16, state_dict)

# 32*4 CUs
state_dict = torch.load(get_latest_model(save_path, 32, 4, QP, 'params'))
remove_state_dict_prefix(state_dict, '1.')
safe_load_state_dict(sub_net_32_4, state_dict)

# 16*4 CUs
state_dict = torch.load(get_latest_model(save_path, 16, 4, QP, 'params'))
remove_state_dict_prefix(state_dict, '1.')
safe_load_state_dict(sub_net_16_4, state_dict)

# 8*4 CUs
state_dict = torch.load(get_latest_model(save_path, 8, 4, QP, 'params'))
remove_state_dict_prefix(state_dict, '1.')
safe_load_state_dict(sub_net_8_4, state_dict)

# 4*32 CUs
state_dict = torch.load(get_latest_model(save_path, 4, 32, QP, 'params'))
remove_state_dict_prefix(state_dict, '1.')
safe_load_state_dict(sub_net_4_32, state_dict)

# 4*16 CUs
state_dict = torch.load(get_latest_model(save_path, 4, 16, QP, 'params'))
remove_state_dict_prefix(state_dict, '1.')
safe_load_state_dict(sub_net_4_16, state_dict)

# 4*8 CUs
state_dict = torch.load(get_latest_model(save_path, 4, 8, QP, 'params'))
remove_state_dict_prefix(state_dict, '1.')
safe_load_state_dict(sub_net_4_8, state_dict)


fid_yuv = open(file_yuv, 'rb')
fid_out = open('CU_partition.dat', 'wb')

#print(QP)

POC = 0
for i_frame in range(n_frame):
    time_pre = time.time()
    frame_YUV = read_YUV420_frame(fid_yuv, width, height)

    if i_frame % intra_period == 0 or i_frame == n_frame - 1:
        #print(QP)
        frame_Y = frame_YUV._Y # shape: [height, width]
        frame_CTUs, CTU_info = mat_to_patches(frame_Y, 64)

        n_samples = np.shape(frame_CTUs)[0]
        x = torch.Tensor(frame_CTUs).to(device)
        qp = torch.Tensor(np.ones(1,)*QP).to(device)
        #print(QP)
        #print(qp)

        f64, f32, f16, f8, f4 = main_net(x)
        f64 = f64.cpu().detach().numpy()
        f32 = f32.cpu().detach().numpy()
        f16 = f16.cpu().detach().numpy()
        f8 = f8.cpu().detach().numpy()
        f4 = f4.cpu().detach().numpy()

        n_CTU_finished = 0
        n_CTU_total = np.shape(CTU_info)[0]

        CU_info_finish = []
        while n_CTU_finished < n_CTU_total:
            i_CTU_start = n_CTU_finished
            i_CTU_end = min(n_CTU_finished + MAX_INPUT_CTUS, n_CTU_total)
            #print([i_CTU_start, i_CTU_end])
            # format of CU_info: [POC, i_CTU, x_start, y_start, width, height, prob_0, prob_1, prob_2, prob_3, prob_4, prob_5]
            #     an example: [87, 10, 192, 64, 64, 64, -1, -1, -1, -1, -1, -1]
            CU_info_remain = []
            load_CU_info(POC, i_CTU_start, CTU_info[i_CTU_start : i_CTU_end, :], CU_info_remain)
            while len(CU_info_remain) > 0:
                update_CU_info(f64, f32, f16, f8, f4, qp, CU_info_remain, CU_info_finish)
                #print('%d CUs predicted. %d CUs remained.' % (len(CU_info_finish), len(CU_info_remain)))
            n_CTU_finished = i_CTU_end

        for CU_info in CU_info_finish:
            fid_out.write(np.array(CU_info).astype(np.int16))

        POC += 1
        time_post = time.time()
        print('%s: Frame %d / %d, time = %.3f' % (file_yuv, i_frame + 1, n_frame, time_post - time_pre))

fid_yuv.close()
fid_out.close()


