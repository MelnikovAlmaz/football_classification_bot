import torch
from torchvision.transforms import transforms

from classifier.networks.basic_net import BasicNet
from classifier.networks.refferi_net import RefferiNet


class Classifier:
    def __init__(self, base_net_path, refferi_net_path, device):
        self.basic_net = BasicNet()
        self.basic_net.load_state_dict(torch.load(base_net_path, map_location=device))

        self.refferi_net = RefferiNet()
        self.refferi_net.load_state_dict(torch.load(refferi_net_path, map_location=device))

        self.transform = transforms.Compose(
            [transforms.Resize((60, 30)),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def preprocess(self, image):
        tensor = self.transform(image)
        tensor.unsqueeze_(0)
        return tensor

    def map_basic_class(self, class_id):
        basic_classes = [
            'blue_team',
            'green_goalkeeper',
            'other',
            'refferi',
            'white_team',
            'yellow_goalkeeper'
        ]
        return basic_classes[class_id]

    def map_refferi_class(self, class_id):
        basic_classes = [
            'main_refferi',
            'side_refferi'
        ]
        return basic_classes[class_id]

    def predict(self, image):
        tensor = self.preprocess(image)

        # Basic class prediction
        outputs = self.basic_net.forward(tensor)
        _, predicted = torch.max(outputs, 1)
        label = self.map_basic_class(predicted)

        if label == 'refferi':
            outputs = self.refferi_net.forward(tensor)
            _, predicted = torch.max(outputs, 1)
            label = self.map_refferi_class(predicted)
        return label