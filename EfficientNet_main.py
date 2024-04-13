import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from Data.dataset import BrainTumorDataset  # 确保路径正确
from EfficientNet import initialize_model  # 确保模型导入正确
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def evaluate_model(model, dataloader, device, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())

            preds = torch.sigmoid(outputs).data > 0.5
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data.unsqueeze(1))

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = running_corrects.double() / len(dataloader.dataset)
    return epoch_loss, epoch_accuracy

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_classes = 1
    num_epochs = 3
    batch_size = 32
    learning_rate = 0.001

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_dir = 'D:\\Assessment\\COM668\\Br35H\\Data'
    datasets = {x: BrainTumorDataset(data_dir, transform=transform, split=x) for x in ['train', 'val', 'test']}
    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val', 'test']}

    model = initialize_model(num_classes)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.unsqueeze(1).float())

                    preds = torch.sigmoid(outputs).data > 0.5
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data.unsqueeze(1))

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # 在测试集上评估模型
    test_loss, test_acc = evaluate_model(model, dataloaders['test'], device, criterion)
    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')

    torch.save(model.state_dict(), 'brain_tumor_classifier.pth')

if __name__ == '__main__':
    main()



