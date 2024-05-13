import os
import json
import re


def load_secrets(filename='secrets.json'):
    if not os.path.exists(filename):
        return
    with open(filename) as f:
        secrets = json.load(f)
        for key, value in secrets.items():
            os.environ[key] = str(value)


def load_profile(profile_name, base_directory="./profiles"):
    # first check if the profile exists
    if not os.path.exists(base_directory):
        # try one directory up
        base_directory = os.path.join("..", base_directory)
        if not os.path.exists(base_directory):
            raise FileNotFoundError(f"Profiles directory '{base_directory}' not found.")
    # Convert the provided profile name to lowercase
    profile_name_lower = profile_name.lower()

    # Get all files in the profiles directory
    files = os.listdir(base_directory)

    # Find the file that matches the profile name (case-insensitive)
    matching_file = next((f for f in files if f.lower() == f"{profile_name_lower}.json"), None)

    if not matching_file:
        raise FileNotFoundError(f"Profile '{profile_name}' not found.")

    file_path = os.path.join(base_directory, matching_file)

    with open(file_path, 'r') as f:
        data = json.load(f)

    return data


def load_config_file(filename='config.ini'):
    import configparser
    config = configparser.ConfigParser()
    config.read(filename)
    return config


def create_config_file(filename='config.ini'):
    import configparser
    config = configparser.ConfigParser()
    config['Avatar'] = {'profile_name': 'Sophia'}
    config['Audio'] = {'input_audio_device': 'Default', 'output_audio_device': 'Default'}
    with open(filename, 'w') as f:
        config.write(f)
    return config

def remove_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)