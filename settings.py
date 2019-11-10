import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TOKEN = '1048782943:AAHbBHvilDGf4avKKzKv4lYv95CYR6FhDlI'

# Model paths
BASIC_NET_PATH = os.path.join(BASE_DIR, 'classifier/models/basic_model.pth')
REFFERI_NET_PATH = os.path.join(BASE_DIR, 'classifier/models/refferi_model.pth')
WHITE_NET_PATH = os.path.join(BASE_DIR, 'classifier/models/white_model.pth')
BLUE_NET_PATH = os.path.join(BASE_DIR, 'classifier/models/blue_model.pth')

DEVICE = 'cpu'
