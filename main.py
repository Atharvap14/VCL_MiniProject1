import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torchvision
from torch import nn

# Load the pre-trained ResNet-18 model
model = torchvision.models.resnet18(pretrained=False)
model.fc = nn.Sequential(
    nn.Dropout(0.5),  # Add dropout with a dropout rate of 0.5
    nn.Linear(model.fc.in_features, 5)
)
model.load_state_dict(torch.load('best_model_retrained.pth'))
model.eval()

# Define class labels (modify as per your dataset)
class_labels = ["Cabinet", "Chair", "Sofa", "Table", "WindowFrame"]

# Define transformations for image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_image(image):
    # Preprocess the image
    img = preprocess(image).unsqueeze(0)
    
    # Make a prediction
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
    
    return predicted

def main():
    st.title("Object Recognition using ResNet-18")
    st.write("This app uses a pre-trained ResNet-18 model to recognize objects in images.")

    # Upload image through Streamlit
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Perform object recognition
        prediction = predict_image(image)
        
        st.write("Prediction:")
        st.write(class_labels[prediction.item()])

if __name__ == "__main__":
    main()
