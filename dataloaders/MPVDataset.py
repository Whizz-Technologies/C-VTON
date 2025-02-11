import os
import json

import cv2
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

semantic_cloth_labels = [
    [128, 0, 128],
    [128, 128, 64],
    [128, 128, 192],
    [0, 255, 0],
    [0, 128, 128], # dress
    [128, 128, 128], # something upper?
    
    [0, 0, 0], # bg
    
    [0, 128, 0], # hair
    [0, 64, 0], # left leg?
    [128, 128, 0], # right hand
    [0, 192, 0], # left foot
    [128, 0, 192], # head
    [0, 0, 192], # legs / skirt?
    [0, 64, 128], # skirt?
    [128, 0, 64], # left hand
    [0, 192, 128], # right foot
    [0, 0, 128],
    [0, 128, 64],
    [0, 0, 64],
    [0, 128, 192]
]

cloth_segm_list = [
  [128,0,0],
  [128,128,0],
  [0,128,0]
]


semantic_densepose_labels = [
    [0, 0, 0], #Background
	[105, 105, 105], 
	[85, 107, 47], #Chest
	[139, 69, 19], #Left Hand
	[72, 61, 139], #Right Hand
	[0, 128, 0], #Right Feet
	[154, 205, 50], #Left Feet
	[0, 0, 139], #left thigh back
	[255, 69, 0], #right left back
	[255, 165, 0], #Left Thigh
	[255, 255, 0], #Right Thigh
	[0, 255, 0], #left leg lower back
	[186, 85, 211], #right leg lower back
	[0, 255, 127], #Left Leg Lower front
	[220, 20, 60], #Right Leg Lower front
	[0, 191, 255], #Right Hand Inner Top
	[0, 0, 255], #Left Hand Inner Top
	[216, 191, 216], #Right Hand Outer Top
	[255, 0, 255], #Left Hand Outer Top
	[30, 144, 255], #Right Hand Inner Bottom
	[219, 112, 147], #Left Hand Inner Bottom
	[240, 230, 140], #Right Hand Outter Bottom
	[255, 20, 147], #Left Hand Outer Bottom
	[255, 160, 122], #Left Head
	[127, 255, 212] #Right Head
]

semantic_body_labels = [
    [127, 127, 127],
    [0, 255, 255],
    [255, 255, 0],
    [127, 127, 0],
    [255, 127, 127],
    [0, 255, 0],
    [0, 0, 0],
    [255, 127, 0],
    [0, 0, 255],
    [127, 255, 127],
    [0, 127, 255],
    [127, 0, 255],
    [255, 255, 127],
    [255, 0, 0],
    [255, 0, 255]
]


class MPVDataset(Dataset):
    
    def __init__(self, opt, phase, test_pairs=None):
        
        opt.label_nc = [len(semantic_body_labels), len(semantic_cloth_labels), len(semantic_densepose_labels)]
        opt.semantic_nc = [label_nc + 1 for label_nc in opt.label_nc]
        
        opt.offsets = [0]
        segmentation_modes = ["body", "cloth", "densepose"]
        for i, mode in enumerate(segmentation_modes):
            if mode in opt.segmentation:
                opt.offsets.append(opt.offsets[-1] + opt.semantic_nc[i] + 1) # to account for extra real/fake class in discriminator output
            else:
                opt.offsets.append(0)
        
        if isinstance(opt.img_size, int):
            # opt.img_size = (opt.img_size, int(opt.img_size * (160 / 256)))
            opt.img_size = (opt.img_size, int(opt.img_size * (192 / 256)))
        
        self.opt = opt
        self.phase = phase
        self.db_path = opt.dataroot
            
        self.filepath_df = pd.read_csv(os.path.join(self.db_path, "all_poseA_poseB_clothes.txt"), sep="\t", names=["poseA", "poseB", "target", "split"])
        self.filepath_df = self.filepath_df.drop_duplicates("poseA")
        self.filepath_df = self.filepath_df[self.filepath_df["poseA"].str.contains("front")]
        self.filepath_df = self.filepath_df.drop(["poseB"], axis=1)
        self.filepath_df = self.filepath_df.sort_values("poseA")

        if phase in {"test", "test_same"}:
            self.filepath_df = self.filepath_df[self.filepath_df.split == "test"]
            
            if phase == "test":
                
                if test_pairs is None:
                    del self.filepath_df["target"]
                    filepath_df_new = pd.read_csv(os.path.join(self.db_path, "test_unpaired_images.txt"), sep=" ", names=["poseA", "target"])
                    self.filepath_df = pd.merge(self.filepath_df, filepath_df_new, how="left")
                else:
                    self.filepath_df = pd.read_csv(os.path.join(self.db_path, test_pairs), sep=" ", names=["poseA", "target"])
            
        elif phase in {"train", "val", "train_whole"}:
            self.filepath_df = self.filepath_df[self.filepath_df.split == "train"]
            
            if phase == "train":
                self.filepath_df = self.filepath_df.iloc[:int(len(self.filepath_df) * opt.train_size)]
            elif phase == "val":
                self.filepath_df = self.filepath_df.iloc[-int(len(self.filepath_df) * opt.val_size):]
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(opt.img_size),
            transforms.ToTensor(),
        ])
        
        if phase in {"train", "train_whole"} and self.opt.add_pd_loss:
            # check if this also works with MPV
            self.hand_indices = [2, 7, 11, 14]
            self.body_label_centroids = [None] * len(self.filepath_df)
        else:
            self.body_label_centroids = None
    
    def __getitem__(self, index):
        df_row = self.filepath_df.iloc[index]

        # get original image of person
        image = cv2.imread(os.path.join(self.db_path, df_row["poseA"]))
        id_img =  df_row["poseA"].split('_')[1].split('.')[0]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        original_size = image.shape[:2]
        
        # extract non-warped cloth
        cloth_image = cv2.imread(os.path.join(self.db_path, df_row["target"]))
        cloth_image = cv2.cvtColor(cloth_image, cv2.COLOR_BGR2RGB)
        cloth_image = cv2.resize(cloth_image,(192,256))
        use_cloth_seg = True
        if use_cloth_seg == True:
          # load cloth labels
          #cloth_seg = cv2.imread(os.path.join(self.db_path, df_row["poseA"][:-4] + "_densepose.png"))
          cloth_seg = cv2.imread('/content/img_t_{}_generated.png'.format(id_img))
          cloth_seg = cv2.cvtColor(cloth_seg, cv2.COLOR_BGR2RGB)
          cloth_seg = cv2.resize(cloth_seg, self.opt.img_size[::-1], interpolation=cv2.INTER_NEAREST)
          
          # mask the image to get desired inputs
          
          # get the mask without upper clothes / dress, hands, neck
          # additionally, get cloth segmentations by cloth part
          cloth_seg_transf = np.zeros(self.opt.img_size)
          mask = np.zeros(self.opt.img_size)
          for i, color in enumerate(cloth_segm_list):
              cloth_seg_transf[np.all(cloth_seg == color, axis=-1)] = i
              if 0 <= i < (2 + self.opt.no_bg):     # this works, because colors are sorted in a specific way with background being the 8th.
                  print("self.opt.no_bg",self.opt.no_bg)
                  mask[np.all(cloth_seg == color, axis=-1)] = 1.0
        else:
          # load cloth labels
          #cloth_seg = cv2.imread(os.path.join(self.db_path, df_row["poseA"][:-4] + "_densepose.png"))
          cloth_seg = cv2.imread('/content/seg_{}_cvton.png'.format(id_img))
          cloth_seg = cv2.cvtColor(cloth_seg, cv2.COLOR_BGR2RGB)
          cloth_seg = cv2.resize(cloth_seg, self.opt.img_size[::-1], interpolation=cv2.INTER_NEAREST)
          
          # mask the image to get desired inputs
          
          # get the mask without upper clothes / dress, hands, neck
          # additionally, get cloth segmentations by cloth part
          cloth_seg_transf = np.zeros(self.opt.img_size)
          mask = np.zeros(self.opt.img_size)
          for i, color in enumerate(semantic_densepose_labels):
              cloth_seg_transf[np.all(cloth_seg == color, axis=-1)] = i
              if 2 <= i < (5 + self.opt.no_bg):     # this works, because colors are sorted in a specific way with background being the 8th.
                  print("self.opt.no_bg",self.opt.no_bg)
                  mask[np.all(cloth_seg == color, axis=-1)] = 1.0
        cv2.imwrite("/content/mask.jpg",mask*255)
        cloth_seg_transf = np.expand_dims(cloth_seg_transf, 0)
        cloth_seg_transf = torch.tensor(cloth_seg_transf)
        
        mask = np.repeat(np.expand_dims(mask, -1), 3, axis=-1).astype(np.uint8)
        masked_image = image * (1 - mask)
        
        # load and process the body labels
        # body_seg = cv2.imread(os.path.join(self.db_path, df_row["poseA"][:-4] + "_densepose.png"))
        body_seg = cv2.imread('/content/seg_{}_cvton.png'.format(id_img))
        body_seg = cv2.cvtColor(body_seg, cv2.COLOR_BGR2RGB)
        body_seg = cv2.resize(body_seg, self.opt.img_size[::-1], interpolation=cv2.INTER_NEAREST)
        
        body_seg_transf = np.zeros(self.opt.img_size)
        for i, color in enumerate(semantic_body_labels):
            body_seg_transf[np.all(body_seg == color, axis=-1)] = i
            
            # additionally, get body segmentation centroids.
            if self.phase == "train" and self.opt.add_pd_loss and (self.body_label_centroids[index] is None or len(self.body_label_centroids[index]) != len(self.hand_indices)) and i in self.hand_indices:
                if self.body_label_centroids[index] is None:
                    self.body_label_centroids[index] = []
                    
                non_zero = np.nonzero(np.all(body_seg == color, axis=-1))
                if len(non_zero[0]):
                    x = int(non_zero[0].mean())
                else:
                    x = -1
                    
                if len(non_zero[1]):
                    y = int(non_zero[1].mean())
                else:
                    y = -1
                    
                self.body_label_centroids[index].append([x, y])
                
        body_label_centroid = self.body_label_centroids[index] if self.body_label_centroids is not None else ""
        
        body_seg_transf = np.expand_dims(body_seg_transf, 0)
        body_seg_transf = torch.tensor(body_seg_transf)
        
        # load and process denspose labels
        # densepose_seg = cv2.imread(os.path.join(self.db_path, df_row["poseA"][:-4] + "_densepose.png"))
        densepose_seg = cv2.imread('/content/seg_{}_cvton.png'.format(id_img))
        densepose_seg = cv2.cvtColor(densepose_seg, cv2.COLOR_BGR2RGB)
        densepose_seg = cv2.resize(densepose_seg, self.opt.img_size[::-1], interpolation=cv2.INTER_NEAREST)
        
        densepose_seg_transf = np.zeros(self.opt.img_size)
        for i, color in enumerate(semantic_densepose_labels):
            densepose_seg_transf[np.all(densepose_seg == color, axis=-1)] = i
            
        densepose_seg_transf = np.expand_dims(densepose_seg_transf, 0)
        densepose_seg_transf = torch.tensor(densepose_seg_transf)
        
        # scale the inputs to range [-1, 1]
        image = self.transform(image)
        image = (image - 0.5) / 0.5
        masked_image = self.transform(masked_image)
        masked_image = (masked_image - 0.5) / 0.5
        cloth_image = self.transform(cloth_image)
        cloth_image = (cloth_image - 0.5) / 0.5
        
        if self.opt.bpgm_id.find("old") >= 0:
            print("OLD")
            # load pose points
            pose_name = df_row["poseA"].replace('.jpg', '_keypoints.json')
            with open(os.path.join(self.db_path, pose_name), 'r') as f:
                try:
                    pose_label = json.load(f)
                    pose_data = pose_label['people'][0]['pose_keypoints_2d']
                    pose_data = np.array(pose_data)
                    pose_data = pose_data.reshape((-1,3))
                
                except IndexError:
                    pose_data = np.zeros((25, 3))

            pose_data[:, 0] = pose_data[:, 0] * (self.opt.img_size[0] / 1024)
            pose_data[:, 1] = pose_data[:, 1] * (self.opt.img_size[1] / 768)
            
            point_num = pose_data.shape[0]
            pose_map = torch.zeros(point_num, *self.opt.img_size)
            r = 5
            im_pose = Image.new('L', self.opt.img_size)
            pose_draw = ImageDraw.Draw(im_pose)
            for i in range(point_num):
                one_map = Image.new('L', self.opt.img_size)
                draw = ImageDraw.Draw(one_map)
                pointx = pose_data[i,0]
                pointy = pose_data[i,1]
                if pointx > 1 and pointy > 1:
                    draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                    pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                
                one_map = self.transform(np.array(one_map))
                pose_map[i] = one_map[0]
                   
            # save background-person mask
            shape = torch.tensor(1 - np.all(cloth_seg == [0, 0, 0], axis=2).astype(np.float32)) * 2 - 1
            shape = shape.unsqueeze(0)
            
            # extract just the head image
            # head_label_colors = [0, 128, 0], [128, 0, 192]
            head_label_colors = [255, 160, 122], [127, 255, 212]
            
            head_mask = torch.zeros(self.opt.img_size)
            for color in head_label_colors:
                head_mask += np.all(cloth_seg == color, axis=2)
            
            im_h = image * head_mask
                
            # cloth-agnostic representation
            agnostic = torch.cat([shape, im_h, pose_map], 0).float()
        else:
            print("New")
            agnostic = ""

        return {"image": {"I": image,
                          "C_t": cloth_image,
                          "I_m": masked_image },
                "cloth_label": {},
                "body_label": {},
                "densepose_label": densepose_seg_transf,
                "name": df_row["poseA"],
                "agnostic": agnostic,
                "original_size": original_size,
                "label_centroid": body_label_centroid}
            
    
    def __len__(self):
        return len(self.filepath_df)
    
    def name(self):
        return "MPVDataset"