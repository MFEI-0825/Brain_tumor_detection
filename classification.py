from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms
from model import BrainTumorCNN

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = BrainTumorCNN().to(device)
model.load_state_dict(torch.load('brain_tumor_model.pth', map_location=device))
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def predict_and_display_image(image_path):
    # Open the image and apply the transform
    image = Image.open(image_path).convert('RGB')
    transformed_image = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(transformed_image)
        probability = torch.sigmoid(output).item()

    # Predicted class
    predicted_class = 'yes' if probability > 0.5 else 'no'

    # Draw the result on the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()  # Loads a default font
    text = f'Pred: {predicted_class}, Prob: {probability:.4f}'
    # Position for the text
    text_position = (10, 10)
    # Add a rectangle behind text for better visibility
    text_background = draw.textbbox(text_position, text, font=font)
    draw.rectangle(text_background, fill='white')
    draw.text(text_position, text, fill='black', font=font)

    # Display the image
    image.show()
    # Save the image if you want to keep the result
    image.save('test_result.jpg')

# Test image path
test_image_path = 'Data/pred/pred2.jpg'  # Replace with the path of your test image

# Predict and display the image
predict_and_display_image(test_image_path)
