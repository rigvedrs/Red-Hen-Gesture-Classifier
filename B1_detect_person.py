import numpy as np
import cv2
import imageio
import pickle
import argparse
import torch




def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_4points(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [x0, y0, x1, y1, x0, y1, x1, y0]
    return torch.stack(b, dim=-1)


def box_4points_to_xyxy(x):
    x0, y0, x1, y1, x2, y2, x3, y3 = x.unbind(-1)
    xs = torch.stack([x0, x1, x2, x3], dim=-1)
    ys = torch.stack([y0, y1, y2, y3], dim=-1)

    b = [
        torch.min(xs, dim=-1).values,
        torch.min(ys, dim=-1).values,
        torch.max(xs, dim=-1).values,
        torch.max(ys, dim=-1).values,
    ]
    return torch.stack(b, dim=-1)


def point_to_abs(points, size):

    original_shape = points.shape
    points = points.reshape(-1, original_shape[-1])
    points, meta = points[:, :2], points[:, 2:]
    points = points * torch.as_tensor([size[1], size[0]], device=points.device)

    transformed_points = torch.cat([points, meta], dim=-1)
    return transformed_points.reshape(original_shape)


def point_to_rel(points, size):

    original_shape = points.shape
    points = points.reshape(-1, original_shape[-1])
    points, meta = points[:, :2], points[:, 2:]
    points = points / torch.as_tensor([size[1], size[0]], device=points.device)

    transformed_points = torch.cat([points, meta], dim=-1)
    return transformed_points.reshape(original_shape)


def points_transformation(points, transformation):
    original_shape = points.shape

    # prepare points [N,3]
    points = points.reshape(-1, original_shape[-1])
    points, meta = points[:, :2], points[:, 2:]
    points = torch.cat([points, torch.as_tensor([[1.0]] * points.shape[0], device=points.device)], dim=1)
    points = points.unsqueeze(2)

    # prepare transformation [N,3,3]
    if len(transformation.shape) == 2:
        transformation = torch.unsqueeze(transformation, dim=0).expand(points.shape[0], 3, 3)

    transformed_points = transformation @ points
    transformed_points = torch.cat([torch.squeeze(transformed_points, 2)[:, :2], meta], dim=-1)
    return transformed_points.reshape(original_shape)


def boxes_to_abs(boxes, size):

    original_shape = boxes.shape
    boxes = boxes.reshape(-1, original_shape[-1])
    boxes, meta = boxes[:, :4], boxes[:, 4:]
    boxes = boxes * torch.as_tensor([size[1], size[0], size[1], size[0]], device=boxes.device)

    transformed_boxes = torch.cat([boxes, meta], dim=-1)
    return transformed_boxes.reshape(original_shape)


def boxes_to_rel(boxes, size):

    original_shape = boxes.shape
    boxes = boxes.reshape(-1, original_shape[-1])
    boxes, meta = boxes[:, :4], boxes[:, 4:]
    boxes = boxes / torch.as_tensor([size[1], size[0], size[1], size[0]], device=boxes.device)

    transformed_boxes = torch.cat([boxes, meta], dim=-1)
    return transformed_boxes.reshape(original_shape)


def boxes_transformation(boxes, transformation):
    original_shape = boxes.shape

    # prepare points [N,3]
    points = boxes.reshape(-1, original_shape[-1])
    # should be possible with a single reshape
    points_xyxy, meta = points[:, :4], points[:, 4:]
    # we need to compute all 4 points

    points = box_xyxy_to_4points(points_xyxy)
    points = points.reshape(-1, 2)
    transformed_points = points_transformation(points, transformation)
    transformed_points = transformed_points.reshape(-1, 8)

    transformed_points = box_4points_to_xyxy(transformed_points)

    transformed_points = torch.cat([transformed_points, meta], dim=-1)
    return transformed_points.reshape(original_shape)


def boxes_fit_size(boxes, size):
    h, w = size[0], size[1]

    original_shape = boxes.shape

    max_size = torch.as_tensor([w, h], dtype=torch.float32, device=size.device)
    boxes = torch.min(boxes.reshape(-1, 2, 2), max_size)
    boxes = boxes.clamp(min=0)

    return boxes.reshape(original_shape)


def boxes_scale(boxes, scale, size=None):

    box_cxcywh = box_xyxy_to_cxcywh(boxes)
    scaled_box_wh = box_cxcywh[2:] * scale
    scaled_box = box_cxcywh_to_xyxy(torch.cat([box_cxcywh[:2], scaled_box_wh], dim=0))
    if size is not None:
        scaled_box = boxes_fit_size(scaled_box, size)

    return scaled_box


def boxes_aspect_ratio(boxes, aspect_ratio, size=None):
    box_cxcywh = box_xyxy_to_cxcywh(boxes)
    w, h = box_cxcywh[2], box_cxcywh[3]
    n_w, n_h = w, h
    if w > aspect_ratio * h:
        n_h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        n_w = h * aspect_ratio
    scaled_box = box_cxcywh_to_xyxy(torch.stack([box_cxcywh[0], box_cxcywh[1], n_w, n_h], dim=0))
    if size is not None:
        scaled_box = boxes_fit_size(scaled_box, size)
    return scaled_box


def post_process_person_predictions(pred_logits, pred_boxes, targets, threshold=0.1):

    predictions = {"boxes": [], "labels": [], "size": targets["size"], "scores": []}

    batch_size = pred_logits.shape[0]

    label_softmax = torch.softmax(pred_logits, dim=-1)
    top_prediction = label_softmax > threshold
    boxes_pos = top_prediction[..., :-1].nonzero()

    for b in range(batch_size):
        boxes = []
        labels = []
        scores = []
        inv_transformation = torch.linalg.inv(targets["transformation"][b])
        weak_boxes_abs = boxes_to_abs(box_cxcywh_to_xyxy(pred_boxes[b]), size=targets["size"][b])
        boxes_origins_abs = boxes_transformation(weak_boxes_abs, inv_transformation)

        boxes_sample = boxes_pos[boxes_pos[:, 0] == b]

        for box in boxes_sample.unbind(0):
            box_index = box[1]
            box_cls = box[2]
            box_cxcywh = boxes_origins_abs[box_index]
            box_score = label_softmax[b, box_index, box_cls]
            labels.append(box_cls)
            boxes.append(box_cxcywh)
            scores.append(box_score)
        if len(boxes) > 0:
            predictions["boxes"].append(torch.stack(boxes, dim=0))
            predictions["labels"].append(torch.stack(labels, dim=0))
            predictions["scores"].append(torch.stack(scores, dim=0))
        else:
            predictions["boxes"].append(
                torch.zeros(
                    [0, 4],
                    device=label_softmax.device,
                )
            )
            predictions["labels"].append(torch.zeros([0], dtype=torch.int64, device=label_softmax.device))
            predictions["scores"].append(torch.zeros([0], device=label_softmax.device))
    return predictions


def preprocess_image(image):
    # Check the dimensions of the image
    height, width = image.shape[:2]

    # If either the width or height is greater than 500 pixels, resize it while maintaining the aspect ratio
    if width > 500 or height > 500:
        if width > height:
            new_width = 500
            new_height = int(height * (new_width / width))
        else:
            new_height = 500
            new_width = int(width * (new_height / height))

        image = cv2.resize(image, (new_width, new_height))

    return image

def detect_person(img_loc='./img.jpg', output_dir='./data/'):
    print('Loading model...')

    person_model = torch.jit.load('model/popart_semi_bbox_v1_trace.pt')

    # img_loc = args.img_loc

    print(f'loading image from {img_loc}...')


    image = imageio.v2.imread(img_loc)



    print('Detecting persons...')

    image = preprocess_image(image)
    prediction = person_model(torch.from_numpy(image).to('cuda'))


    # targets describes the change of the image before it was given into the model, here everything is left on default.
    targets = {
        "size": [image.shape[0:2]],
        "origin_size": [image.shape[0:2]],
        "transformation": [torch.tensor([[1.0, 0.0, 0.0000], [0.0000, 1.0, 0.0000], [0.0000, 0.0, 1.0000]])],
    }
    final_person_prediciton = post_process_person_predictions(prediction[0], prediction[1], targets=targets)

    all_preds_boxes = []
    all_preds_orig_size = []

    cropped_images = []
    for box in final_person_prediciton["boxes"][0]:
        xyxy = box_cxcywh_to_xyxy(box).detach().numpy()
        box_image = image[max(0,int(box[1])):int(box[3]),max(0,int(box[0])):int(box[2]),:]

        all_preds_boxes.append([max(0,int(box[0])), max(0,int(box[1])), int(box[2]), int(box[3])])


        # fig, axs = plt.subplots(1, 1)
        # axs.imshow(box_image)
        cropped_images.append(box_image)
        

    # Create the list of dictionaries
    image_dicts = []
    for img in cropped_images:
        image_dict = {
            'image': img,
            'shape': img.shape,
            'keypoints': None,
            'pose': None,
            'palm': None,
            'person':None,
            'pred': None
        }
        image_dicts.append(image_dict)

    # Save the list of dictionaries to a file using pickle
    save_dir = f'{output_dir}image_data.pickle'

    print(f'Saving image dictionaries in {save_dir}...')

    with open(save_dir, 'wb') as file:
        pickle.dump(image_dicts, file)

    print('Succesfully Completed!')


if __name__ == '__main__': 
    # Parsing arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_loc', type=str, default='./img.jpg', help='File path to the content image')
    # parser.add_argument('--num', type=int, default=0, help='Number of images to download, 0 for all, and any other number for otherwise')
    parser.add_argument('--output_dir', type=str, default='./data/', help='File path to output directory')

    args = parser.parse_args()

    img_loc = args.img_loc

    output_dir = args.output_dir

    detect_person(img_loc, output_dir)