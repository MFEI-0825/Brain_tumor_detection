import os
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, random_split

class BrainTumorDataset(Dataset):
    def __init__(self, data_dir, transform=None, split='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        self.image_paths, self.labels = self._get_image_paths_and_labels()
        self._split_dataset()

    def _get_image_paths_and_labels(self):
        image_paths = []
        labels = []
        for category in ['no', 'yes']:
            class_num = 1 if category == 'yes' else 0
            class_path = os.path.join(self.data_dir, category)
            for file in os.listdir(class_path):
                if file.endswith('.jpg') or file.endswith('.png'):
                    image_paths.append(os.path.join(class_path, file))
                    labels.append(class_num)
        combined = list(zip(image_paths, labels))
        random.shuffle(combined)
        image_paths, labels = zip(*combined)
        return image_paths, labels

    def _split_dataset(self):
        # 假设训练：验证：测试 = 70% : 15% : 15%
        total_images = len(self.image_paths)
        train_size = int(0.7 * total_images)
        val_size = int(0.15 * total_images)
        test_size = total_images - train_size - val_size

        # 通过split属性确定使用哪个数据子集
        if self.split == 'train':
            self.image_paths = self.image_paths[:train_size]
            self.labels = self.labels[:train_size]
        elif self.split == 'val':
            self.image_paths = self.image_paths[train_size:train_size + val_size]
            self.labels = self.labels[train_size:train_size + val_size]
        elif self.split == 'test':
            self.image_paths = self.image_paths[train_size + val_size:]
            self.labels = self.labels[train_size + val_size:]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# 数据预处理和转换
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 设置数据集路径
data_dir = 'D:\\Assessment\\COM668\\Br35H\\Data'

# 创建数据集实例
train_dataset = BrainTumorDataset(data_dir, transform=transform, split='train')
val_dataset = BrainTumorDataset(data_dir, transform=transform, split='val')
test_dataset = BrainTumorDataset(data_dir, transform=transform, split='test')

