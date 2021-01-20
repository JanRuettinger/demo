import timm
import torch
import os
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image, ImageFile
import numpy as np
import pandas as pd
import ast
from pathlib import Path

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_model(model_name, checkpoint_path, num_classes=7178):
    models_from_timm = ['tf_efficientnet_b7_ns','mixnet_l', 'pnasnet5large','inception_resnet_v2', 'dpn98'] # 'tresnet_xl' doesn't work
    models_from_pytorch = ['resnext101_32x8d', 'alexnet']
    valid_model_names = models_from_timm + models_from_pytorch

    assert(model_name in valid_model_names), "Your selected model is not supported."

    if model_name in models_from_timm:
      model = timm.create_model(model_name,num_classes=num_classes,in_chans=3)
    else:
      if model_name == 'resnext101_32x8d':
          model = torchvision.models.resnext101_32x8d()
          model.fc =  nn.Linear(in_features=2048, out_features=num_classes, bias=True)
      elif model_name == 'alexnet':
          model = torchvision.models.alexnet()
          model.classifier[6] = nn.Linear(4096,num_classes)
    print("Model pretrained with ImgNet was loaded.")

    print('Resuming from checkpoint ...')
    assert os.path.isfile(checkpoint_path), 'Error: no checkpoint found!'
    checkpoint = torch.load(checkpoint_path)

    # self.config = checkpoint["trainer_config"]
    # print(self.config)
    state_dict = checkpoint['state_dict']

    print("Load model state dict ...")
    model.load_state_dict(state_dict)

    return model




class OpenImagesDataset(Dataset):
    """Open Images Dataset V4"""
    # Download link: https://github.com/cvdfoundation/open-images-dataset

    def __init__(self, mode, image_dir_path, data_label_path,num_classes):
        """
        Args:
            mode (string): Specifies if the dataset should is a train/validation/test dataset. Possible values: "train", "validation", "test"
            data_label_path (string): Path to json file containing information about dataset (image_id, labels)
            image_dir_path (string): Path to directory with images.
            data_augmentatio (boolean, optional): Specifies if data augmentation should be used or not
            num_classes (int, optional): Number of classes
        """

        self.mode = mode # train || validation || test
        try:
            self.data_df = pd.read_json(data_label_path)  # Read the json file which maps image_id to labels
        except:
            print(data_label_path)
            raise Exception("Sorry, mate")
        self.image_dir_path = image_dir_path
        self.num_classes = num_classes
        self.transform = None # here a transform function will be injected later through the call of timm.data.create_loader()

    def __len__(self):
        return len(self.data_df.index) # calculate length of dataset

    def __getitem__(self, index):

        # Get image name from the pandas df
        
        row = self.data_df.iloc[index]
        single_image_name = row.ImageID
        single_image_path = Path(self.image_dir_path) / \
            f'{single_image_name}.jpg'

        # Open image
        img_as_np = Image.open(single_image_path).convert("RGB")
        
        if self.transform is not None: # will not be NONE during runtime!
            img_as_tensor = self.transform(img_as_np).float()

        if self.mode == "train" or self.mode == "validation":

            image_label = np.array([0.0]*self.num_classes) # empty labels array
            image_labels = ast.literal_eval(row.ClassID) # array containing indices of lables which are true e.g. [3,28,229]
                    
            for label in image_labels: # assign 1 to all true labels => result image_labels is K-hot encoded label vector
                image_label[label] = 1.0

            image_label_as_tensor = torch.from_numpy(image_label).float()
            return img_as_tensor, image_label_as_tensor
        
        if self.mode == "test": # no ground truth for test images available
            return img_as_tensor, single_image_name