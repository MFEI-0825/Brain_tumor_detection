import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from Data.dataset import BrainTumorDataset
from model import initialize_model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Set device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Hyperparameters
num_epochs = 15
batch_size = 32
learning_rate = 0.001

# Data transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Data paths
data_dir = 'D:\\Assessment\\COM668\\Br35H\\Data'

# Load data
dataset = BrainTumorDataset(data_dir, transform=transform)
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
model, optimizer, loss_fn = initialize_model()
model.to(device)

# Training the model
def train_model():
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        with tqdm(train_loader, unit="batch") as tepoch:
            for images, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                images = images.to(device)
                labels = labels.type(torch.FloatTensor).unsqueeze(1).to(device)  # Change the labels to FloatTensor

                # Forward pass
                outputs = model(images)
                loss = loss_fn(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {total_loss / len(train_loader):.4f}')

# Testing the model and saving it
def test_and_save_model():
    model.eval()
    total = 0
    correct = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.type(torch.FloatTensor).unsqueeze(1).to(device)
            outputs = model(images)
            predicted = (outputs.data > 0.5).float()  # Apply threshold
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.view(-1).cpu().numpy())
            y_pred.extend(predicted.view(-1).cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Accuracy on test images: {accuracy}%')
    print(f'Accuracy score: {accuracy_score(y_true, y_pred)}')

    # Save the trained model
    torch.save(model.state_dict(), 'brain_tumor_model.pth')
    print("Saved trained model.")

if __name__ == '__main__':
    train_model()
    test_and_save_model()


