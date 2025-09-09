
def load_and_predict_48x48_grayscale(model_path, image_path):
    """
    Load the enhanced model and predict emotion from 48x48 grayscale image.
    
    Args:
        model_path (str): Path to saved model
        image_path (str): Path to 48x48 grayscale image
        
    Returns:
        tuple: (predicted_emotion, confidence, all_probabilities)
    """
    import torch
    from PIL import Image
    from torchvision import transforms
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    model = create_enhanced_grayscale_model(
        num_classes=checkpoint['model_config']['num_classes'],
        dropout_rate=checkpoint['model_config']['dropout_rate'],
        use_transfer_adaptation=False,  # Already trained
        device='cpu'
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('L').resize((48, 48))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image_tensor = transform(image).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Convert to emotion name
    label_map = checkpoint['label_map']
    reverse_map = {v: k for k, v in label_map.items()}
    predicted_emotion = reverse_map[predicted_class]
    
    return predicted_emotion, confidence, probabilities[0].numpy()
