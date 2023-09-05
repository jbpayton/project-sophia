import sys

from langchain.tools import tool
from tools.StableDiffusionImageGenerator import StableDiffusionImageGenerator


@tool("image_generator_tool", return_direct=False)
def image_generator_tool(prompt: str) -> str:
    """This tool allows an agent to provide a prompt and generate an image. Please be descriptive as to what the
    subject is wearing, their setting, things like eye and hair color, and their action. The image filename is
    returned. """

    try:
        output = StableDiffusionImageGenerator().generate_image(prompt)

        # After it is finished the tool should return a string that is the output of the tool.
        return output
    except:
        # print the exception
        return str(sys.exc_info())