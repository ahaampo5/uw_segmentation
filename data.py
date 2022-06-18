import numpy as np
import torch
from torch.utils.data import DataLoader
import albumentations as A 
import cv2


def load_img(path):
    img = np.load(path)
    img = img.astype('float32') # original is uint16
    mx = np.max(img)
    if mx:
        img/=mx # scale image to [0, 1]
    return img

def load_msk(path):
    msk = np.load(path)
    msk = msk.astype('float32')
    msk/=255.0
    return msk


def get_train_transforms(CFG):
    if CFG.hard_arg:
        return A.Compose([
        # A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
        A.RandomRotate90(),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=10, p=0.9, border_mode=cv2.BORDER_REFLECT),
        A.OneOf([
            # A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
            # A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
            # A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            A.ElasticTransform(p=.3),
            A.GaussianBlur(p=.3),
            A.GaussNoise(p=.3),
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
            # A.IAAPiecewiseAffine(p=0.3),
        ], p=0.3),
        A.OneOf([
            A.HueSaturationValue(15,25,0),
            A.CLAHE(clip_limit=2),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
        ], p=0.3),
        A.CoarseDropout(max_holes=8, max_height=CFG.img_size[0]//20, max_width=CFG.img_size[1]//20,
                         min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
        A.Normalize(),
        ], p=1.0)
    else:
        return A.Compose([
            # A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
    #         A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
    # #             A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            ], p=0.25),
            A.CoarseDropout(max_holes=8, max_height=CFG.img_size[0]//20, max_width=CFG.img_size[1]//20,
                             min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
            ], p=1.0)

def get_valid_transforms(CFG):
    return A.Compose([
        # A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
        ], p=1.0)


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, CFG, df, label=True, transforms=None):
        self.df         = df
        self.CFG        = CFG
        self.label      = label
        self.img_paths  = df['image_path'].tolist()
        self.msk_paths  = df['mask_path'].tolist()
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path  = self.img_paths[index]
        img = []
        img = load_img(img_path)
        if self.CFG.normalize == 'imagenet':
            img = img * 255.
            img = img.astype('uint8')
        
        if self.label:
            msk_path = self.msk_paths[index]
            msk = load_msk(msk_path)
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img  = data['image']
                msk  = data['mask']
            img = np.transpose(img, (2, 0, 1))
            msk = np.transpose(msk, (2, 0, 1))
            return torch.tensor(img), torch.tensor(msk)
        else:
            if self.transforms:
                data = self.transforms(image=img)
                img  = data['image']
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img)
    
    
def prepare_loaders(fold, df, train_transforms, valid_transforms, CFG, debug=False):
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    if CFG.full_train:
        train_df = df.copy()
    valid_df = df.query("fold==@fold").reset_index(drop=True)
    if debug:
        train_df = train_df.head(32*5).query("empty==0")
        valid_df = valid_df.head(32*3).query("empty==0")
    train_dataset = BuildDataset(CFG, train_df, transforms=train_transforms)
    valid_dataset = BuildDataset(CFG, valid_df, transforms=valid_transforms)

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size if not debug else 20, 
                              num_workers=4, shuffle=True, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size*2 if not debug else 20, 
                              num_workers=4, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader