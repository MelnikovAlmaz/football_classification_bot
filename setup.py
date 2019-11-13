import os
import telebot
from flask import Flask, request

import settings

from classifier.classifier import Classifier
from classifier.data.image_processing import image_from_file

bot = telebot.TeleBot(settings.TOKEN,)
server = Flask(__name__)

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

@server.route('/' + settings.TOKEN, methods=['POST'])
def getMessage():
    bot.process_new_updates([telebot.types.Update.de_json(request.stream.read().decode("utf-8"))])
    return "!", 200


@server.route("/")
def webhook():
    bot.remove_webhook()
    bot.set_webhook(url='https://fbumabot.herokuapp.com/' + settings.TOKEN)
    return "!", 200


if __name__ == "__main__":
    server.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))


