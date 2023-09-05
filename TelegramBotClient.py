import os
import telebot
import util
from Agents import get_avatar_agent

util.load_secrets()
BOT_TOKEN = os.environ.get('BOT_TOKEN')
TELEGRAM_CHAT_ID = int(os.environ.get('TELEGRAM_CHAT_ID'))

bot = telebot.TeleBot(BOT_TOKEN)

# read profile name and input/output audio device from a config file
# if not specified, use the default values
config = util.load_config_file()

# if the config file is not found, (avatar not found in profile) create a new one
if 'Avatar' not in config:
    config = util.create_config_file()

# initialize agent
profile_name = config['Avatar']['profile_name']

profile = util.load_profile(profile_name)

avatar = get_avatar_agent(profile)


class IsAllowedUser(telebot.custom_filters.SimpleCustomFilter):
    # Class will check whether the user is admin or creator in group or not
    key = 'is_allowed_user'

    @staticmethod
    def check(message: telebot.types.Message):
        return message.from_user.id in [TELEGRAM_CHAT_ID]


# To register filter, you need to use method add_custom_filter.
bot.add_custom_filter(IsAllowedUser())


@bot.message_handler(commands=['start', 'hello'], is_allowed_user=True)
def send_welcome(message):
    avatar.receive(message.from_user.first_name, "Hello!")
    avatar_response = avatar.send()
    bot.send_message(message.chat.id, avatar_response)


@bot.message_handler(func=lambda msg: True, is_allowed_user=True)
def echo_all(message):
    avatar.receive(message.from_user.first_name, message.text)

    while True:
        bot.send_chat_action(chat_id=message.chat.id, action="typing")
        avatar_response = avatar.send()
        bot.send_message(message.chat.id, avatar_response)
        if avatar.has_picture_to_show:
            bot.send_photo(message.chat.id, photo=open(avatar.image_to_show, 'rb'))
            avatar.has_picture_to_show = False
            avatar.needs_to_think_more = False
            break
        if not avatar.needs_to_think_more:
            break
        bot.send_chat_action(chat_id=message.chat.id, action="typing")




print("Bot is running...")
bot.infinity_polling()
