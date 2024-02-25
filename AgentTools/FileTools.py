import os


def SaveFile(filename, contents):
    """Writes text to a file"""
    with open(filename, 'w') as file:
        file.write(contents)
        return "Successfully wrote text to file."


def LoadFile(filename):
    """Reads text from a file"""
    with open(filename, 'r') as file:
        return file.read()


def ListFilesInWorkingDirectory():
    """Lists all files in the working directory"""
    return os.listdir()