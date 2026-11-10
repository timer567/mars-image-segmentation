import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "train_ready")


class SegmentationDataset(Dataset):
    """
    图像分割数据集类，用于加载图像和对应的分割掩码。
    
    参数:
        data_list_file (str): 包含文件名列表的文本文件路径（不含扩展名）
        image_dir (str): 图像文件所在目录路径
        mask_dir (str): 掩码文件所在目录路径
        mode (str): 数据集模式，'train' 或 'val'
        base_size (int): 基础尺寸，用于 resize
        crop_size (int): 随机裁剪尺寸
        mean (list): 归一化均值 [R, G, B]
        std (list): 归一化标准差 [R, G, B]
    """
    
    def __init__(self, data_list_file, image_dir, mask_dir, mode='train', 
                 base_size=518, crop_size=518, 
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        # 验证模式参数
        assert mode in ['train', 'val'], "mode 必须是 'train' 或 'val'"
        self.mode = mode
        
        # 存储路径信息
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        
        # 尺寸参数
        self.base_size = base_size
        self.crop_size = crop_size
        
        # 归一化参数
        self.mean = mean
        self.std = std
        
        # 读取文件列表
        with open(data_list_file, 'r') as f:
            self.file_list = [line.strip() for line in f if line.strip()]
        
        # 验证文件列表不为空
        assert len(self.file_list) > 0, f"文件列表为空: {data_list_file}"
        
        # 定义基础变换（适用于 train 和 val）
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.file_list)
    
    def _resize(self, image, mask, size):
        """统一调整图像和掩码的尺寸"""
        image = image.resize((size, size), Image.BILINEAR)
        mask = mask.resize((size, size), Image.NEAREST)  # 掩码使用最近邻插值保持标签值不变
        return image, mask
    
    def _random_crop(self, image, mask):
        """随机裁剪图像和掩码到 crop_size x crop_size"""
        w, h = image.size
        if w < self.crop_size or h < self.crop_size:
            # 如果图像小于裁剪尺寸，则先填充再裁剪
            pad_w = max(0, self.crop_size - w)
            pad_h = max(0, self.crop_size - h)
            image = transforms.Pad((0, 0, pad_w, pad_h), fill=0)(image)
            mask = transforms.Pad((0, 0, pad_w, pad_h), fill=0)(mask)
            w, h = image.size
            
        # 随机选择裁剪起始点
        left = torch.randint(0, w - self.crop_size + 1, (1,)).item()
        top = torch.randint(0, h - self.crop_size + 1, (1,)).item()
        
        image = image.crop((left, top, left + self.crop_size, top + self.crop_size))
        mask = mask.crop((left, top, left + self.crop_size, top + self.crop_size))
        return image, mask
    
    def _horizontal_flip(self, image, mask, prob=0.5):
        """以一定概率进行水平翻转"""
        if torch.rand(1) < prob:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        return image, mask
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        返回:
            image (Tensor): 形状为 (3, H, W) 的归一化图像张量
            mask (Tensor): 形状为 (H, W) 的整数类型掩码张量，值域为 [0, 9]
        """
        # 获取文件名（不含扩展名）
        filename = self.file_list[idx]
        
        # 构建完整路径
        image_path = os.path.join(self.image_dir, filename + '.png')  # 图像也是 png 格式
        mask_path = os.path.join(self.mask_dir, filename + '.png')   # 掩码是 png 格式
        
        # 加载图像和掩码
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path)
        
        # 确保掩码是单通道灰度图
        if mask.mode != 'L':
            mask = mask.convert('L')
        
        # 转换掩码为 numpy 数组并确保值域正确
        mask_np = np.array(mask)
        # 确保掩码值在 0-9 范围内
        assert mask_np.min() >= 0 and mask_np.max() <= 9, f"掩码值超出范围 [0,9]: {filename}"
        
        # 调整到基础尺寸
        image, mask = self._resize(image, mask, self.base_size)
        
        if self.mode == 'train':
            # 训练模式：应用随机裁剪和水平翻转
            image, mask = self._random_crop(image, mask)
            image, mask = self._horizontal_flip(image, mask)
        else:
            # 验证模式：中心裁剪或直接使用 resize 后的图像
            # 这里我们直接使用 resize 后的图像（因为 base_size == crop_size）
            pass
        
        # 转换为张量并归一化
        image = self.base_transform(image)
        
        # 掩码转换为长整型张量
        mask = torch.from_numpy(np.array(mask)).long()
        
        return image, mask


# 使用示例和测试代码
if __name__ == "__main__":
    # 数据路径配置
    DATA_ROOT = DEFAULT_DATA_ROOT
    IMAGE_DIR = os.path.join(DATA_ROOT, "images")
    MASK_DIR = os.path.join(DATA_ROOT, "masks")
    
    # 创建训练数据集
    train_dataset = SegmentationDataset(
        data_list_file=os.path.join(DATA_ROOT, "train.txt"),
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        mode='train',
        base_size=518,
        crop_size=518
    )
    
    # 创建验证数据集
    val_dataset = SegmentationDataset(
        data_list_file=os.path.join(DATA_ROOT, "val.txt"),
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        mode='val',
        base_size=518,
        crop_size=518
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 测试获取一个样本
    image, mask = train_dataset[0]
    print(f"图像张量形状: {image.shape}")  # 应为 torch.Size([3, 518, 518])
    print(f"掩码张量形状: {mask.shape}")    # 应为 torch.Size([518, 518])
    print(f"图像张量数据类型: {image.dtype}")  # 应为 torch.float32
    print(f"掩码张量数据类型: {mask.dtype}")   # 应为 torch.int64 (long)
    print(f"掩码唯一值: {torch.unique(mask)}") # 应在 0-9 范围内
    
    # 验证数据集测试
    val_image, val_mask = val_dataset[0]
    print(f"验证图像张量形状: {val_image.shape}")  # 应为 torch.Size([3, 518, 518])
    print(f"验证掩码张量形状: {val_mask.shape}")   # 应为 torch.Size([518, 518])
