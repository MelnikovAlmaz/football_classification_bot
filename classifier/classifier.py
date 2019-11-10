import torch
from torchvision.transforms import transforms

from classifier.networks.basic_net import BasicNet
from classifier.networks.player_net import PlayerNet
from classifier.networks.refferi_net import RefferiNet


class Classifier:
    def __init__(self, base_net_path, refferi_net_path, white_net_path, blue_net_path,  device):
        self.basic_net = BasicNet()
        self.basic_net.load_state_dict(torch.load(base_net_path, map_location=device))

        self.refferi_net = RefferiNet()
        self.refferi_net.load_state_dict(torch.load(refferi_net_path, map_location=device))

        self.white_net = PlayerNet()
        self.white_net.load_state_dict(torch.load(white_net_path, map_location=device))

        self.blue_net = PlayerNet()
        self.blue_net.load_state_dict(torch.load(blue_net_path, map_location=device))

        self.transform = transforms.Compose(
            [transforms.Resize((60, 30)),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def preprocess(self, image):
        """
        Transform Pillow image with random size to tensor with size (60, 30)
        :param image: Pillow image
        :return: torch Tensor with size [1, 60, 30, 3]
        """
        tensor = self.transform(image)
        tensor.unsqueeze_(0)
        return tensor

    def predict(self, image):
        # Transform Pillow image to standardized Tensor
        tensor = self.preprocess(image)

        # Basic class prediction
        outputs = self.basic_net.forward(tensor)
        _, predicted = torch.max(outputs, 1)
        label = self.map_basic_label(predicted)

        # Classify refferies to main and side
        if label == 'refferi':
            outputs = self.refferi_net.forward(tensor)
            _, predicted = torch.max(outputs, 1)
            label = self.map_refferi_label(predicted)
        # Classify white team players
        elif label == 'white_team':
            outputs = self.white_net.forward(tensor)
            _, predicted = torch.max(outputs, 1)
            label = self.map_white_team_label(predicted)
        # Classify blue team players
        elif label == 'blue_team':
            outputs = self.white_net.forward(tensor)
            _, predicted = torch.max(outputs, 1)
            label = self.map_white_team_label(predicted)

        return label

    def map_basic_label(self, class_id):
        labels = [
            'blue_team',
            '23',  # green_goalkeeper'
            '24',  # other
            'refferi',
            'white_team',
            '3'  # yellow_goalkeeper
        ]
        return labels[class_id]

    def map_refferi_label(self, class_id):
        labels = ['8', '20']
        return labels[class_id]

    def map_white_team_label(self, class_id):
        labels = ['12', '13', '15', '16', '17', '18', '19', '21', '4', '7']
        return labels[class_id]

    def map_blue_team_label(self, class_id):
        labels = ['0', '1', '10', '11', '14', '2', '22', '5', '6', '9']
        return labels[class_id]
