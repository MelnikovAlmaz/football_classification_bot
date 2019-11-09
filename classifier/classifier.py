class Classifier:
    def __init__(self):
        self.basic_classifier = BasicClassifier()

        self.transform = transforms.Compose(
            [transforms.Resize((60, 30)),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def preprocess(image):
        tensor = self.transform(image)
        tensor.unsqueeze_(0)
        return tensor

    def map_class(class_id):
        basic_classes = [
            'blue_team',
            'green_goalkeeper',
            'other',
            'refferi',
            'white_team',
            'yellow_goalkeeper'
        ]
        return basic_classes[class_id]

    def predict(image):
        tensor = self.preprocess(image)
        basic_class_id = self.basic_classifier(tensor)
        basic_class_label = self.map_class(basic_class_id)
        return basic_class_label