from torch.utils.data import DataSet

class ChestDataset(Dataset):
    def __init__(self, df, transform, mode='train'):
        self.label_df = df[all_labels].values
        self.path_df = df['path'].values
        self.transform = transform
        self.mode = mode
        
    def __len__(self):
        return len(self.path_df)
    
    def __getitem__(self, idx):
        path = self.path_df[idx]
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        if self.mode=='val':
            img = cv2.resize(img, (CROP_SIZE, CROP_SIZE))
            
        img = np.stack([img,img,img], -1)
        
        if self.transform:
            img = self.transform(img)
            

        label = self.label_df[idx]
            

        return img, torch.tensor(label, dtype=torch.float)