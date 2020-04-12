import os
import os.path
from torch.utils.data import Dataset


class flickr15k_dataset_lite(Dataset):
    def __init__(self, root='./static/images'):
        self.root = root
        self.gt_path = os.path.join(self.root, 'groundtruth')
        self.img_set_path = os.path.join(self.root, 'dataset')
        self.gt = {}

        for i in range(1, 34):
            """flickr15k has 33 classes, silly way"""
            self.gt[str(i)] = []
        file = open(self.gt_path)
        for line in file:
            sketch_cls = line.split()[0]
            img_path = line.split()[1][:-4] + '.jpg'
            img_cls = img_path.split('/')[0]
            img_name = img_path.split('/')[1][:-4]
            img_path = os.path.join(self.img_set_path, img_path)
            # check img exist
            if os.path.exists(img_path):
                self.gt[sketch_cls].append((img_path, img_cls, img_name))
        file.close()

        self.datapath = []
        for i in range(1, 34):
            item = str(i)
            for fn in self.gt[item]:
                # item: class number
                # f[0]: file absolute path
                # f[1]: class name
                # f[2]: file name
                self.datapath.append((fn[1], item, fn[0], fn[2]))

    def __getitem__(self, idx):
        photo = self.datapath[idx]
        return photo

    def __len__(self):
        return len(self.datapath)
