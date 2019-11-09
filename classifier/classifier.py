import torch
from torchvision.transforms import transforms

from classifier.networks.basic_classifier import BasicNet


class Classifier:
    def __init__(self, base_net_path, device):
        self.basic_net = BasicNet()
        self.basic_net.load_state_dict(torch.load(base_net_path, map_location=device))

        self.transform = transforms.Compose(
            [transforms.Resize((60, 30)),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def preprocess(self, image):
        tensor = self.transform(image)
        tensor.unsqueeze_(0)
        return tensor

    def map_class(self, class_id):
        basic_classes = [
            'blue_team',
            'green_goalkeeper',
            'other',
            'refferi',
            'white_team',
            'yellow_goalkeeper'
        ]
        return basic_classes[class_id]

    def predict(self, image):
        tensor = self.preprocess(image)
        outputs = self.basic_net.forward(tensor)
        _, predicted = torch.max(outputs, 1)
        basic_class_label = self.map_class(predicted)
        return basic_class_label