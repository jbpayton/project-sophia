import os

def list_objects(obj_path="./AgentStorage"):
    # Get all the files in the directory
    obj_filenames = os.listdir(obj_path)
    # load all filenames that end with .hobj but do not store the extension
    obj_names = [obj_filename[:-5] for obj_filename in obj_filenames if obj_filename.endswith(".hobj")]
    return obj_names


def store_object_to_file(obj, obj_name, obj_path="./AgentStorage"):
    # Write the object to a file
    with open(obj_path + "/" + obj_name + ".hobj", "wb") as file:
        file.write(obj)


def load_object_from_file(obj_name, obj_path="./AgentStorage"):
    # Read the object from a file
    with open(obj_path + "/" + obj_name + ".hobj", "rb") as file:
        return file.read()


def delete_object_from_file(obj_name, obj_path="./AgentStorage"):
    # Delete the object from a file
    os.remove(obj_path + "/" + obj_name + ".hobj")


def make_box(directory_name, base_path="./AgentStorage"):
    # Make a directory
    directory_path = base_path + "/" + directory_name
    os.mkdir(directory_path)


def delete_box(directory_name, base_path="./AgentStorage"):
    # Delete a directory
    directory_path = base_path + "/" + directory_name
    os.rmdir(directory_path)


def list_boxes(base_path="./AgentStorage"):
    # List all directories
    return os.listdir(base_path)


def list_objects_in_box(directory_name, base_path="./AgentStorage"):
    # List all files in a directory under the base path only
    directory_path = base_path + "/" + directory_name
    return os.listdir(directory_path)


def move_object_to_box(obj_name, directory_name, base_path="./AgentStorage"):
    # Move a file to a directory
    directory_path = base_path + "/" + directory_name
    obj_path = base_path + "/" + obj_name + ".hobj"
    os.rename(obj_path, directory_path + "/" + obj_name + ".hobj")


def move_object_from_box(obj_name, directory_name, base_path="./AgentStorage"):
    # Move a file from a directory
    directory_path = base_path + "/" + directory_name + "/" + obj_name + ".hobj"
    obj_path = base_path + "/" + obj_name + ".hobj"
    os.rename(directory_path, obj_path)

