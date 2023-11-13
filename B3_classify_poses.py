import torch
import torch.nn as nn
import pickle
import argparse
import ultralytics
from ultralytics import YOLO

import matplotlib.pyplot as plt
import matplotlib.image as mpimg  


def display_image(img, pred):
    # img = mpimg.imread(file_path)
    plt.figure(figsize=(10, 10))
    plt.title(pred)
    plt.imshow(img)
    plt.axis('off')
    plt.show()



# Define the network architecture
class PoseClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(PoseClassifier, self).__init__()
        # Input layer
        self.input = nn.Linear(input_size, hidden_size)
        # Hidden layers
        self.hidden1 = nn.Linear(hidden_size, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.hidden3 = nn.Linear(128, hidden_size)
        self.hidden4 = nn.Linear(hidden_size, hidden_size)
        # Output layer
        self.output = nn.Linear(hidden_size, num_classes)
        # Activation function
        self.relu = nn.ReLU()
        # Softmax function
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        # Pass the input through the network
        x = self.input(x)
        x = self.relu(x)
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.hidden3(x)
        x = self.relu(x)
        x = self.hidden4(x)
        x = self.relu(x)
        x = self.output(x)
        return x
    


def classify_poses(file_dir='data/image_data.pickle', output_dir='./data/'):
    print('Loading pickle file...')

    # Load the data back and restore the images
    with open(file_dir, 'rb') as file:
        image_dicts = pickle.load(file)
        
    print('Pickle file successfully loaded')



    # Define the hyperparameters
    input_size = 12# Number of input features (17 keypoints * 2 coordinates)
    hidden_size = 64 # Number of hidden units in each layer
    num_classes = 3 # Number of pose classes

    # Loading the model
    c_model = PoseClassifier(input_size, hidden_size, num_classes)  
    c_model.load_state_dict(torch.load('model/c_model_V2_2.pt'))



    for i, d in enumerate(image_dicts):
        kp = d['keypoints'][5:11]
        inp = []
        for x,y in kp:
            inp.append(x/384)
            inp.append(y/288)
            
        inp = torch.tensor(inp)
        pred = int(torch.argmax(c_model(inp)))
        # print(pred)
        
        image_dicts[i]['pose'] = pred
        
    print('Loading palm detection model...')

	

	# Detecting the palm gestures using YOLO
    model = YOLO('model/palm.pt') 

    print('Detecting Gestures...')

    for i, image_dict in enumerate(image_dicts):
        image = image_dict['image']
        out = model(image)
        
        if out[0].boxes.cls.size()[0]:
            image_dicts[i]['palm'] = out[0].names[int(out[0].boxes.cls[0].cpu())]
			
			
			
	# Classifying the persons using the YOLO classifying model		
    model = YOLO('model/person_classifier.pt') 

    print('Detecting persons...')

    for i, image_dict in enumerate(image_dicts):
        image = image_dict['image']
        out = model(image)

        if out[0].probs:
            image_dicts[i]['person'] = out[0].names[out[0].probs.top1]
            
            
  
        

    for d in image_dicts:
        img = d['image']
        if d['pose'] == 0:
            if d['person'] == 'mary':
                out = d['pred'] = 'Mary with both Hands Raised'
            else:
                out = d['pred'] = 'Both Hands Raised'
            print(f' Detected {out}')
            display_image(img, d['pred'])
        
        elif d['pose'] == 1 and d['palm'] == 'prayer':
            
            
            if d['person'] == 'mary':
                out = d['pred'] = 'Mary in prayer Gesture'
            else:
                out = d['pred'] = 'Prayer Gesture'
            
            print(f' Detected {out}')
            display_image(img, d['pred'])
            
        # elif d['pose'] == 2:
        #     d['pred'] = 'Hands Crossed on chest'
        #     display_image(img, d['pred'])
            
        elif d['palm'] == 'blessing':

            if d['person'] == 'baby jesus':
                out = d['pred'] = 'Baby Jesus using blessing Gesture'
            elif d['person'] == 'baby jesus mary':
                out = d['pred'] = 'Mary is holding Baby Jesus who is using the blessing gesture'
            else:
                out = d['pred'] = 'Blessing Gesture'
            print(f' Detected {out}')
            display_image(img, d['pred'])
            
        else:
            continue
            

    # Save the list of dictionaries to a file using pickle
    save_dir = f'{output_dir}image_data.pickle'

    print(f'Saving image dictionaries in {save_dir}...')

    with open(save_dir, 'wb') as file:
        pickle.dump(image_dicts, file)
        
    print('Pickle file successfully saved')

    print('Successfully Completed')


if __name__=='__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_dir', type=str, default= 'data/image_data.pickle', help='Path to pickle file')
    parser.add_argument('--output_dir', type=str, default='./data/', help='File path to output directory')

    args = parser.parse_args()

    classify_poses(args.file_dir, args.output_dir)

