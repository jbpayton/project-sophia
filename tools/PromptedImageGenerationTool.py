import sys

from langchain.tools import tool
from tools.StableDiffusionImageGenerator import StableDiffusionImageGenerator


@tool("image_generator_tool", return_direct=False)
def image_generator_tool(subject: str, action: str, setting: str) -> str:
    """This tool allows an agent to provide a prompt and generate an image. Please be descriptive as to what the
    subject is wearing, their setting, things like eye and hair color, and their action. The image filename is
    returned. Please provide the filename to the agent enclose in brackets, like this: [
    generated_images/image-filename.png], this will allow the agent to send the image to the user.

    Here are some examples of the parameters, the subject example is tuned to look like Sophia, feel free to modify it:
    subject = '1girl, 1girl, in 20s, anime, petite, short hair, bob cut, blue hair, ' \
                  'detailed large eyes, sparkling eyes, black sweater, long sleeves, black jeans,'
    action = 'looking at viewer'
    setting = 'in futuristic library'

    """

    try:
        style = "(intricate details), (****),"
        prompt = subject + ", " + action + ", " + style + ", " + setting
        output = StableDiffusionImageGenerator().generate_image(prompt)

        # After it is finished the tool should return a string that is the output of the tool.
        return output
    except:
        # print the exception
        return str(sys.exc_info())