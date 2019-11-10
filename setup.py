import telebot

import settings

from classifier.classifier import Classifier
from classifier.data.image_processing import image_from_file

bot = telebot.TeleBot(settings.TOKEN,)

# Init image classifier
classifier = Classifier(
    base_net_path=settings.BASIC_NET_PATH,
    refferi_net_path=settings.REFFERI_NET_PATH,
    white_net_path=settings.WHITE_NET_PATH,
    blue_net_path=settings.BLUE_NET_PATH,
    device=settings.DEVICE
)


@bot.message_handler(content_types=['photo'])
def get_photo_message(message):
    """
    Predict label of request photos.
    :param message: massage that contains photo
    :return: label of photo, string
    """
    # Download photo and save as file object
    telegram_file_id = message.photo[-1].file_id
    telegram_file = bot.get_file(telegram_file_id)
    photo_file = bot.download_file(telegram_file.file_path)

    # Convert photo file to Pillow Image
    image = image_from_file(photo_file)
    # Predict label of image
    class_label = classifier.predict(image)
    # Send label as answer to chat
    bot.send_message(message.chat.id, str(class_label))


# Start pooling messages from bot
bot.polling(none_stop=True, timeout=90)

