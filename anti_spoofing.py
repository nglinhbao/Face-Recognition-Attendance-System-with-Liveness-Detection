import os
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from anti_spoofing_dir.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE
from anti_spoofing_dir import transform as trans
from anti_spoofing_dir.utility import get_kernel, parse_model_name

# Define model mapping
MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE': MiniFASNetV1SE,
    'MiniFASNetV2SE': MiniFASNetV2SE
}

MODEL_PATH = 'anti_spoofing_dir/2.7_80x80_MiniFASNetV2.pth'

class AntiSpoofPredict:
    def __init__(self, device_id):
        self.device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")
        self.model = None

    def _load_model(self):
        # Define the model
        model_name = os.path.basename(MODEL_PATH)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        self.kernel_size = get_kernel(h_input, w_input)
        self.model = MODEL_MAPPING[model_type](conv6_kernel=self.kernel_size).to(self.device)

        # Load model weights
        state_dict = torch.load(MODEL_PATH, map_location=self.device)
        if list(state_dict.keys())[0].startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, img):
        self._load_model()
        test_transform = trans.Compose([trans.ToTensor()])
        img = test_transform(img)
        img = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            result = self.model(img)
            result = F.softmax(result, dim=1).cpu().numpy()
        return result

def spoofing_detection(img, device_id=0):
    # # Convert the image array to a PIL image
    # img = Image.fromarray((image_array * 255).astype('uint8'))
    
    # # Initialize the prediction model
    predictor = AntiSpoofPredict(device_id)
    prediction = predictor.predict(img)
    
    # Assuming the model outputs a probability distribution, use the second class probability (spoof)
    is_spoof = prediction[0][1] > 0.5  # Adjust threshold if needed
    return is_spoof