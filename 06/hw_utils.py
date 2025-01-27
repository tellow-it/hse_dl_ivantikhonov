import torch


def intersection_over_union(boxes_preds, boxes_labels, box_format='midpoint'):
    """
    Calculates intersection over union
    
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes are (x,y,w,h) or (x1,y1,x2,y2) respectively.
    
    Returns:
        tensor: Intersection over union for all examples
    """
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0] - boxes_preds[..., 2] / 2
        box1_y1 = boxes_preds[..., 1] - boxes_preds[..., 3] / 2
        box1_x2 = boxes_preds[..., 0] + boxes_preds[..., 2] / 2
        box1_y2 = boxes_preds[..., 1] + boxes_preds[..., 3] / 2

        box2_x1 = boxes_labels[..., 0] - boxes_labels[..., 2] / 2
        box2_y1 = boxes_labels[..., 1] - boxes_labels[..., 3] / 2
        box2_x2 = boxes_labels[..., 0] + boxes_labels[..., 2] / 2
        box2_y2 = boxes_labels[..., 1] + boxes_labels[..., 3] / 2
    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0]
        box1_y1 = boxes_preds[..., 1]
        box1_x2 = boxes_preds[..., 2]
        box1_y2 = boxes_preds[..., 3]

        box2_x1 = boxes_labels[..., 0]
        box2_y1 = boxes_labels[..., 1]
        box2_x2 = boxes_labels[..., 2]
        box2_y2 = boxes_labels[..., 3]
    else:
        raise ValueError(f"Box format: {box_format} not supported!, try midpoint/corners")

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    union = box1_area + box2_area - intersection

    iou = torch.where(union > 0, intersection / (union + 1e-6), torch.tensor(0.0))

    return iou


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Does Non Max Suppression given bboxes
    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes
    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """
    if len(bboxes) == 0:
        return []

    bboxes = [bbox for bbox in bboxes if bbox[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    nms_bboxes = []

    while bboxes:
        highest_prob_bbox = bboxes.pop(0)
        nms_bboxes.append(highest_prob_bbox)

        bboxes = [
            bbox
            for bbox in bboxes
            if bbox[0] != highest_prob_bbox[0]
               or intersection_over_union(
                torch.tensor(highest_prob_bbox[2:]),
                torch.tensor(bbox[2:]),
                box_format=box_format,
            ) < iou_threshold
        ]

    return nms_bboxes


def mean_average_precision(
        pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Calculates mean average precision 
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bbox
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """
    # List to store average precision for each class
    average_precisions = []

    # Iterate over all classes
    for c in range(num_classes):
        detections = [box for box in pred_boxes if box[1] == c]
        ground_truths = [box for box in true_boxes if box[1] == c]

        # Dictionary to count the number of ground truths per image
        amount_bboxes = {}
        for gt in ground_truths:
            img_idx = gt[0]
            if img_idx not in amount_bboxes:
                amount_bboxes[img_idx] = 0
            amount_bboxes[img_idx] += 1

        # Store which ground truth boxes were already used for evaluation
        for img_idx in amount_bboxes:
            amount_bboxes[img_idx] = torch.zeros(amount_bboxes[img_idx])

        # Sort detections by confidence score
        detections.sort(key=lambda x: x[2], reverse=True)

        TP = torch.zeros(len(detections))  # True positives
        FP = torch.zeros(len(detections))  # False positives
        total_true_bboxes = len(ground_truths)

        # Iterate through all detections
        for detection_idx, detection in enumerate(detections):
            img_ground_truths = [
                gt for gt in ground_truths if gt[0] == detection[0]
            ]

            best_iou = 0
            best_gt_idx = -1

            # Find the ground truth with the highest IoU
            for idx, gt in enumerate(img_ground_truths):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # True positive, mark this ground truth as used
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    # False positive, this ground truth was already used
                    FP[detection_idx] = 1
            else:
                # False positive, IoU is below threshold
                FP[detection_idx] = 1

        # Compute cumulative sums of TP and FP
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        # Compute precision and recall
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-6)
        recalls = TP_cumsum / (total_true_bboxes + 1e-6)

        # Add a (0, 1) point for recall and precision for completeness
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        # Calculate average precision (AP) using the trapezoidal rule
        average_precision = torch.trapz(precisions, recalls)
        average_precisions.append(average_precision)

    # Compute mean average precision (mAP)
    return sum(average_precisions) / len(average_precisions)
