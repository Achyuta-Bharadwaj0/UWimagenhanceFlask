import os
import torch
from PIL import Image
from os.path import exists, basename
from torchvision.utils import save_image
import torchvision.transforms as transforms
from models.funiegan import GeneratorFunieGAN
from flask import Flask, request, jsonify
import io

app = Flask(__name__)

def enhance_image(input_image):


    # Check the path of trained model
    model_path = os.path.join("trained/FUnIE_GAN/test/generator_95.pth")
    assert exists(model_path), "model weights not found"

    # Set device for pytorch
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')

    # Set data pipeline
    img_width, img_height, channels = 512, 512, 3
    transforms_ = [transforms.Resize((img_height, img_width), transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
    transform = transforms.Compose(transforms_)

    # Initialize generator of FUnIE_GAN
    model = GeneratorFunieGAN().to(DEVICE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()


    inp_img = transform(input_image)
    inp_img = inp_img.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        gen_img = model(inp_img)
        print(type(gen_img))
    # save output image
    save_image(gen_img.data, "enhance.jpg", normalize=True)
    print("done")
    enhanced_image = Image.open("enhance.jpg")  # Placeholder, replace this with your actual code
    return enhanced_image

@app.route('/enhance', methods=['POST'])
def enhance():
    try:
        # Get the image from the request
        image_data = request.files['image'].read()
        input_image = Image.open(io.BytesIO(image_data))

        # Enhance the image
        enhanced_image = enhance_image(input_image)

        # Convert the enhanced image to bytes
        enhanced_image_bytes = io.BytesIO()
        enhanced_image.save(enhanced_image_bytes, format='JPEG')
        enhanced_image_bytes = enhanced_image_bytes.getvalue()

        return jsonify({'enhanced_image': enhanced_image_bytes.decode('latin-1')})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


