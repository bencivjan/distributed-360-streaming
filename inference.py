import torch
from torchvision import transforms
from PIL import Image
import sys

def preprocess_image(image_path, input_size=(224, 224)):
    """
    Preprocess the input image for the model.
    
    Parameters:
    image_path (str): Path to the input image.
    input_size (tuple): Desired input size for the model (default is (224, 224)).
    
    Returns:
    torch.Tensor: Preprocessed image tensor.
    """
    preprocess = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def run_inference(model, image_tensor):
    """
    Run inference on the input image tensor using the loaded model.
    
    Parameters:
    model (torch.nn.Module): Loaded PyTorch model.
    image_tensor (torch.Tensor): Preprocessed image tensor.
    
    Returns:
    torch.Tensor: Model output.
    """
    with torch.no_grad():
        output = model(image_tensor)
    return output

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python inference.py <model_path> <image_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    model = torch.load(model_path)
    if type(model) is dict and 'model' in model:
        model = model['model'].float()
    model.eval()  # Set the model to evaluation mode

    image_tensor = preprocess_image(image_path)
    output = run_inference(model, image_tensor)

    print("Model output:", output)
    output_image = Image.fromarray(output)
    output_image.save('./output_image')