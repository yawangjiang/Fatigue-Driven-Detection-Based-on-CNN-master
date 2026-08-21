import Config
from itertools import product as product
from math import sqrt as sqrt
import torch

# 自动检测设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_prior_box():
    mean_layer = []
    for k, f in enumerate(Config.feature_map):
        mean = []
        for i, j in product(range(f), repeat=2):
            f_k = Config.image_size / Config.steps[k]
            cx = (j + 0.5) / f_k
            cy = (i + 0.5) / f_k

            s_k = Config.sk[k] / Config.image_size
            mean += [cx, cy, s_k, s_k]

            s_k_prime = sqrt(s_k * Config.sk[k + 1] / Config.image_size)
            mean += [cx, cy, s_k_prime, s_k_prime]
            for ar in Config.aspect_ratios[k]:
                mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]

        # --- 【修改 1】：自动适配 CPU/GPU ---
        mean = torch.Tensor(mean).to(device).view(Config.feature_map[k], Config.feature_map[k], -1).contiguous()

        mean.clamp_(max=1, min=0)
        mean_layer.append(mean)

    return mean_layer


def encode(match_boxes, prior_box, variances):
    g_cxcy = (match_boxes[:, :2] + match_boxes[:, 2:]) / 2 - prior_box[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * prior_box[:, 2:])
    # match wh / prior wh
    g_wh = (match_boxes[:, 2:] - match_boxes[:, :2]) / prior_box[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


def change_prior_box(box):
    # --- 【修改 2】：自动适配 CPU/GPU ---
    return torch.cat((box[:, :2] - box[:, 2:] / 2,  # xmin, ymin
                      box[:, :2] + box[:, 2:] / 2), 1).to(device)


# 计算两个box的交集
def insersect(box1, box2):
    label_num = box1.size(0)
    box_num = box2.size(0)
    max_xy = torch.min(
        box1[:, 2:].unsqueeze(1).expand(label_num, box_num, 2),
        box2[:, 2:].unsqueeze(0).expand(label_num, box_num, 2)
    )
    min_xy = torch.max(
        box1[:, :2].unsqueeze(1).expand(label_num, box_num, 2),
        box2[:, :2].unsqueeze(0).expand(label_num, box_num, 2)
    )
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    inter = insersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def point_form(boxes):
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,  # xmin, ymin
                      boxes[:, :2] + boxes[:, 2:] / 2), 1)  # xmax, ymax


def match(threshold, truths, priors, labels, loc_t, conf_t, idx):
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)

    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]  # Shape: [num_priors,4]
    conf = labels[best_truth_idx] + 1  # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    loc = encode(matches, priors, (0.1, 0.2))
    loc_t[idx] = loc  # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def log_sum_exp(x):
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


def decode(loc, priors, variances):
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def nms(boxes, scores, overlap=0.5, top_k=200):
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep, 0
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # store result in iou
        idx = idx[IoU.le(overlap)]
    return keep, count


if __name__ == '__main__':
    mean = default_prior_box()
    print(mean)