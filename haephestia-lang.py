# this is the main file for the haephestia language, it defines a singleton class and functions for the language

from ToolLoader import load_tools_from_files
import StorageLoader

class HaephestiaLangManager:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HaephestiaLangManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self.tools_dict = load_tools_from_files()
            self.objects_list = StorageLoader.list_objects()
            self.box_list = StorageLoader.list_boxes()


def desc_objects_and_boxes():
    haephestia = HaephestiaLangManager()

    # list the objects and boxes in the form of a string that describes them
    objects = "Objects: "
    for obj in haephestia.objects_list:
        objects += obj + ", "
    objects = objects[:-2]

    boxes = "Boxes: "
    for box in haephestia.box_list:
        boxes += box + ", "
    boxes = boxes[:-2]

    return objects + "\n" + boxes
