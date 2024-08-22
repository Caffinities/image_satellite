import torch
import numpy as np
import cv2
from skimage.measure import regionprops
from skimage.morphology import dilation, disk
from scipy.spatial.distance import cdist

def get_instances(mask : np.array,
                  disk_size : int = 0,
                  bbox_size_lim : int = 20,
                  ):
    # using connected component labeling
    selem = disk(disk_size)
    dilated_test = dilation(mask, selem)
    num_labels, labels = cv2.connectedComponents(dilated_test)
    # num_labels, labels = cv2.connectedComponents(mask)
    properties = regionprops(labels)
    instances = []
    index = 0
    for prop in properties:
        bbox = prop.bbox
        bbox_height = bbox[2] - bbox[0]
        bbox_width = bbox[3] - bbox[1]
        if bbox_height <= bbox_size_lim or bbox_width <= bbox_size_lim:
            continue
        # get centroid
        centroid = prop.centroid
        instances.append({
            "label": index,
            "bbox": bbox,
            "centroid": centroid
        })
        index += 1

    return instances

def compute_iou(box0, box1):
    x0_min, y0_min, x0_max, y0_max = box0
    x1_min, y1_min, x1_max, y1_max = box1
    xi0 = max(x0_min, x1_min)
    yi0 = max(y0_min, y1_min)
    xi1 = min(x0_max, x1_max)
    yi1 = min(y0_max, y1_max)
    inter_area = max(0, xi1 - xi0) * max(0, yi1 - yi0)
    box0_area = (x0_max - x0_min) * (y0_max - y0_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    union_area = box0_area + box1_area - inter_area
    iou = inter_area / union_area if union_area != 0 else 0

    return iou

def get_minimum_bounding_box(boxes):
    x_min = min(box[0] for box in boxes)
    y_min = min(box[1] for box in boxes)
    x_max = max(box[2] for box in boxes)
    y_max = max(box[3] for box in boxes)

    return (x_min, y_min, x_max, y_max)

def crop_tensor(tensor, box):
    x_min, y_min, x_max, y_max = box

    return tensor[x_min:x_max, y_min:y_max]

def calculate_iou(tensor0, tensor1):
    assert tensor0.shape == tensor1.shape
    intersection = torch.logical_and(tensor0, tensor1).sum().item()
    union = torch.logical_or(tensor0, tensor1).sum().item()
    iou = intersection / union if union != 0 else 0

    return iou


def get_match(instances_t0,
              instances_t1,
              iou_threshold : float = 0.5,
              distance_threshold : float = 15.0):
    centroids_t0 = np.array([inst['centroid'] for inst in instances_t0])
    centroids_t1 = np.array([inst['centroid'] for inst in instances_t1])
    distance_matrix = cdist(centroids_t0, centroids_t1)
    iou_matrix = np.zeros((len(instances_t0), len(instances_t1)))
    for i, inst0 in enumerate(instances_t0):
        for j, inst1 in enumerate(instances_t1):
            iou_matrix[i, j] = compute_iou(inst0['bbox'], inst1['bbox'])
    matched_pairs = []
    for i, inst0 in enumerate(instances_t0):
        best_match = None
        best_score = float('inf')
        for j, inst1 in enumerate(instances_t1):
            distance = distance_matrix[i, j]
            iou = iou_matrix[i, j]
            if distance < distance_threshold and iou > iou_threshold:
                score = distance - iou 
                # Score to be minimized
                if score < best_score:
                    best_score = score
                    best_match = j
        if best_match is not None:
            matched_pairs.append((inst0, instances_t1[best_match]))
    matched_t0_indices = [pair[0]['label'] for pair in matched_pairs]
    matched_t1_indices = [pair[1]['label'] for pair in matched_pairs]
    unmatched_t0 = [inst for inst in instances_t0 if inst['label'] not in matched_t0_indices]
    unmatched_t1 = [inst for inst in instances_t1 if inst['label'] not in matched_t1_indices]
    
    return matched_t0_indices, matched_t1_indices, unmatched_t0, unmatched_t1, matched_pairs, distance_matrix

def get_nearest_pairs(unmatched_t0,
                      unmatched_t1,
                      distance_matrix,
                      instances_t0,
                      instances_t1):
    nearest_pairs0 = []
    nearest_pairs1 = []
    nn_pairs0_dict = {}
    nn_pairs1_dict = {}
    not_matched_0 = set()
    not_matched_1 = set()
    for i, inst0 in enumerate(unmatched_t0):
        best_score = float('inf')
        for j, inst1 in enumerate(unmatched_t1):
            distance = distance_matrix[inst0['label'], inst1['label']]
            # iou = iou_matrix[i, j]
            if distance < best_score:
                best_score = distance
                best_match = inst1['label']

        if best_score > 80:
            not_matched_0.add(inst0['label'])
        else:
            nearest_pairs0.append((inst0, instances_t1[best_match]))
            nn_pairs0_dict[inst0['label']] = instances_t1[best_match]

    for j, inst1 in enumerate(unmatched_t1):
        best_score = float('inf')
        for i, inst0 in enumerate(unmatched_t0):
            distance = distance_matrix[inst0['label'], inst1['label']]
            # iou = iou_matrix[j, i]
            if distance < best_score:
                best_score = distance
                best_match = inst0['label']
        
        if best_score > 100:
            not_matched_1.add(inst1['label'])
        else:
            nearest_pairs1.append((instances_t0[best_match], inst1))
            nn_pairs1_dict[inst1['label']] = instances_t0[best_match]
    
    return nearest_pairs0, nearest_pairs1, nn_pairs0_dict, nn_pairs1_dict, not_matched_0, not_matched_1

def arrow_count(nearest_pairs0,
                nearest_pairs1):
    inst0_count = {}
    for inst0, inst1 in nearest_pairs1:
        if inst0 == None:
            continue
        label = inst0['label']
        if label in inst0_count:
            inst0_count[label] += 1
        else:
            inst0_count[label] = 1
    inst1_count = {}
    for inst0, inst1 in nearest_pairs0:
        if inst1 == None:
            continue
        label = inst1['label']
        if label in inst1_count:
            inst1_count[label] += 1
        else:
            inst1_count[label] = 1
    multi_targeted_inst1 = [key for key, value in inst1_count.items() if value > 1]
    multi_targeted_inst0 = [key for key, value in inst0_count.items() if value > 1]

    return inst0_count, inst1_count, multi_targeted_inst0, multi_targeted_inst1

def get_change(unmatched_t1,
               distance_matrix,
               multi_targeted_inst0,
               multi_targeted_inst1,
               instances_t0,
               instances_t1,
               nearest_pairs1,
               nn_pairs0_dict,
               nn_pairs1_dict,
               not_matched_1,
               segment0,
               segment1,
               ):
    change_in_1 = set()
    single_in_1 = set([inst['label'] for inst in unmatched_t1])

    for label_1 in multi_targeted_inst1:
        insts = [instances_t0[key] for key, value in nn_pairs0_dict.items() if value['label'] == label_1]
        insts.append(instances_t1[label_1])
        big_box = get_minimum_bounding_box([insts[i]['bbox'] for i in range(len(insts))])
        seg0_crop = crop_tensor(segment0, big_box)
        seg1_crop = crop_tensor(segment1, big_box)
        if calculate_iou(seg0_crop, seg1_crop)<=0.4:
            change_in_1.add(label_1)
        single_in_1.remove(label_1)

    for label_0 in multi_targeted_inst0:
        insts = [instances_t1[key] for key, value in nn_pairs1_dict.items() if value['label'] == label_0]
        labels = set([inst['label'] for inst in insts])
        insts.append(instances_t0[label_0])
        big_box = get_minimum_bounding_box([insts[i]['bbox'] for i in range(len(insts))])
        seg0_crop = crop_tensor(segment0, big_box)
        seg1_crop = crop_tensor(segment1, big_box)
        if calculate_iou(seg0_crop, seg1_crop)<=0.4:
            change_in_1 = change_in_1.union(labels)
        single_in_1 = single_in_1 - labels
    
    single_in_1 = single_in_1 - not_matched_1 #exclude the not matched instances

    for label_1 in single_in_1:
        inst0 = nn_pairs1_dict[label_1]
        inst1 = instances_t1[label_1]
        insts =  [inst0, inst1]
        big_box = get_minimum_bounding_box([insts[i]['bbox'] for i in range(len(insts))])
        seg0_crop = crop_tensor(segment0, big_box)
        seg1_crop = crop_tensor(segment1, big_box)
        # print(calculate_iou(seg0_crop, seg1_crop))
        if calculate_iou(seg0_crop, seg1_crop)<=0.30 or distance_matrix[inst0['label'], label_1]>100:
            change_in_1.add(label_1)

    for inst0, inst1 in nearest_pairs1:
        if inst0 == None:
            change_in_1.add(inst1)
    
    change_in_1 = change_in_1 | not_matched_1
    
    return change_in_1