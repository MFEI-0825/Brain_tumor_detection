import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from Data.dataset import BrainTumorDataset
from EfficientNet import initialize_model
from PIL import Image
import numpy as np

def load_model(path, num_classes):
    # 初始化模型并加载给定路径的权重
    model = initialize_model(num_classes)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_image(model, image_path):
    # 转换图像以匹配训练时的格式
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载图像并应用转换
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 增加批量维度

    # 进行预测
    with torch.no_grad():
        outputs = model(image)
        predicted = torch.sigmoid(outputs).item() > 0.5
        probability = torch.sigmoid(outputs).item()

    return predicted, probability

def main():
    # 模型路径和类别数
    model_path = 'brain_tumor_classifier.pth'
    num_classes = 1  # 肿瘤检测为二分类问题

    # 加载模型
    model = load_model(model_path, num_classes)

    # 图像路径
    test_image_path = 'Data/pred/pred2.jpg'  # 替换为测试图像的路径

    # 使用模型进行预测
    predicted, probability = predict_image(model, test_image_path)
    print(f"Predicted: {'Positive' if predicted else 'Negative'}, Probability: {probability:.4f}")

if __name__ == '__main__':
    main()
