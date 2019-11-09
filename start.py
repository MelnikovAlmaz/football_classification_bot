import telebot

import settings

from classifier.classifier import Classifier
from classifier.data.image_processing import image_from_bytes

bot = telebot.TeleBot(settings.TOKEN,)

classifier = Classifier(settings.BASIC_NET_PATH, settings.DEVICE)


@bot.message_handler(content_types=['photo'])
def get_text_messages(message):
    fileID = message.photo[-1].file_id
    file = bot.get_file(fileID)
    downloaded_file = bot.download_file(file.file_path)
    image = image_from_bytes(downloaded_file)

    class_label = classifier.predict(image)
    bot.send_message(message.chat.id, str(class_label))


bot.polling(none_stop=True, timeout=90)
