import numpy as np
import torch
import pickle
import argparse
import torchvision.transforms as T
import torchvision.transforms.functional as F
from typing import List, Tuple, Any, Optional
import PIL
import random
import math
from scipy.optimize import linear_sum_assignment


def point_to_abs(points, size):

    original_shape = points.shape
    points = points.reshape(-1, original_shape[-1])
    points, meta = points[:, :2], points[:, 2:]
    points = points * torch.as_tensor([size[1], size[0]], device=points.device)

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

def post_process_keypoint_predictions(pred_logits, pred_coords, targets, threshold=0.1):
  translated_coords = []
  for b in range(len(targets["transformation"])):
      # print(b)
      # print(out_coords.shape)
      inv_transformation = torch.linalg.inv(targets["transformation"][b])
      coords_abs = point_to_abs(pred_coords[b], targets["size"][b])
      coords_origin_abs = points_transformation(coords_abs, inv_transformation)
      translated_coords.append(coords_origin_abs)
      # print(coords_origin_abs)
  out_coords = torch.stack(translated_coords, dim=0)

  assert len(pred_logits) == len(targets["size"])
  assert torch.as_tensor(targets["size"]).shape[1] == 2

  num_joints = pred_logits.shape[-1] - 1

  prob = torch.nn.functional.softmax(pred_logits, -1)

  prob_cpu = torch.nn.functional.softmax(pred_logits[..., :-1], dim=-1).detach().cpu()

  _, labels = prob[..., :-1].max(-1)

  scores_list = []
  coords_list = []
  labels_list = []
  for b, C in enumerate(prob_cpu):

      _, query_ind = linear_sum_assignment(-C.transpose(0, 1))  # Cost Matrix: [17, N]
      score = prob_cpu[b, query_ind, list(np.arange(num_joints))].numpy()

      coord = out_coords[b, query_ind].detach().cpu().numpy()
      scores_list.append(torch.as_tensor(score))
      coords_list.append(torch.as_tensor(coord))
      labels_list.append(labels[b, query_ind])
  scores = torch.stack(scores_list)
  coords = torch.stack(coords_list)
  labels = torch.stack(labels_list)

  results = [
      {"scores": s, "labels": l, "keypoints": b, "selected": s > threshold}
      for s, l, b in zip(scores, labels, coords)
  ]

  return results






def detect_keypoints(file_dir='data/image_data.pickle', output_dir='./data/'):
    print('Loading pickle file...')

    # To load the data back and restore the images
    with open(file_dir, 'rb') as file:
        image_dicts = pickle.load(file)

    # Restore the images from the 1D arrays
    for image_dict in image_dicts:
        image_data = image_dict['image']
        image_shape = image_dict['shape']  # Replace with the actual shape of your images
        image_dict['image'] = np.array(image_data).reshape(image_shape)
       

    print('Loading Model')

    keypoint_model = torch.jit.load('model/popart_semi_kpoint_v1_trace.pt')

    print('Detecting Keypoints')


    for i, image_dict in enumerate(image_dicts):
        # print(person_image.shape)
        person_image = image_dict['image']

        mean = [int(255 * x) for x in [0.485, 0.456, 0.406]]
        person_image = np.array(F.resize(F.to_pil_image(person_image), [384,288]))

        prediction = keypoint_model(torch.from_numpy(person_image))

        targets = {
            "size": [person_image.shape[0:2]],
            "origin_size": [person_image.shape[0:2]],
            "transformation": [torch.tensor([[1.0, 0.0, 0.0000], [0.0000, 1.0, 0.0000], [0.0000, 0.0, 1.0000]])],
        }
        # targets describes the change of the image before it was given into the model, here everything is left on default.

        final_keypoint_prediciton = post_process_keypoint_predictions(prediction[0], prediction[1], targets=targets)[0]
        # print(final_keypoint_prediciton)
        
        image_dicts[i]['keypoints'] = final_keypoint_prediciton['keypoints']

        for k, v in final_keypoint_prediciton.items():
            final_keypoint_prediciton[k] = [v.detach()]

    #   plot = plot_prediction_images_cv2(person_image, keypoints=final_keypoint_prediciton["keypoints"], keypoints_labels=final_keypoint_prediciton["labels"], keypoints_scores=final_keypoint_prediciton["scores"])
    #   fig, axs = plt.subplots(1, 1)
    #   axs.imshow(plot)

    # Save the list of dictionaries to a file using pickle
    save_dir = f'{output_dir}image_data.pickle'

    print(f'Saving image dictionaries in {save_dir}...')

    with open(save_dir, 'wb') as file:
        pickle.dump(image_dicts, file)
        
    print('Successfully Completed!')


if __name__ == '__main__': 
    # Parsing arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_dir', type=str, default= 'data/image_data.pickle', help='Path to pickle file')
    parser.add_argument('--output_dir', type=str, default='./data/', help='File path to output directory')

    args = parser.parse_args()

    detect_keypoints(args.file_dir, args.output_dir)