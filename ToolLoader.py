import sys
from inspect import signature, getmembers, isfunction
import importlib.util
import os

# Function to get the dictionary of a tool
def get_tool_name_and_dict(func):
    # Extracting parameter names
    params = list(signature(func).parameters.keys())
    # Extracting the docstring as description

    # If there is no docstring, print to the console and return None
    if not func.__doc__:
        print(f"Warning: {func.__name__} has no docstring. So it will not be loaded as a tool.")
        return None, None
    description = func.__doc__.strip()

    # return the name of the function and the dictionary
    return func.__name__, {"params": params, "description": description, "func": func}


def load_tools_from_files(tool_filenames=None, tool_path="./AgentTools"):
    # create a dictionary to store the tools temporarily
    tools = {}

    if tool_filenames is None:
        # Get all the files in the directory
        tool_filenames = os.listdir(tool_path)
        # Remove the __init__.py file
        tool_filenames = [tool_filename for tool_filename in tool_filenames if tool_filename.endswith(".py") and tool_filename != "__init__.py" and tool_filename != "ToolTemplate.py"]

    for tool_filename in tool_filenames:
        loaded_tools = load_tools_from_file(tool_filename, tool_path)

        # add the tools to the tools dictionary
        tools.update(loaded_tools)

    return tools


def load_tools_from_file(tool_filename, tool_path="./AgentTools"):
    # create a dictionary to store the tools temporarily
    tools = {}

    # Parse the module name from the filename
    module_name = os.path.splitext(os.path.basename(tool_filename))[0]

    # Check if the module is already loaded
    if module_name in sys.modules:
        print("Module already loaded, we are going to reload it")

    # Load the module
    try:
        spec = importlib.util.spec_from_file_location(module_name, tool_path + "/" + tool_filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[module_name] = module
    except Exception as e:
        print(f"Error loading module {module_name}: {e}")
        # return a dictionary with the module name and a detailed error message
        # also a create a function that returns the error message and add it to func
        return {module_name: {"params": [], "description": f"Error loading module {module_name}: {e}", "func": lambda: f"Error loading module {module_name}"}}


    # iterate through all functions in the module
    for name, obj in getmembers(module):
        # if the object is a function
        if isfunction(obj):
            # get the name and dictionary of the function
            name, dictionary = get_tool_name_and_dict(obj)
            # if the dictionary is not None
            if dictionary is not None:
                # store the dictionary in the tools dictionary
                tools[name] = dictionary
                print(name, dictionary)

    return tools

def execute_tool(tool_call_json, tool_dict):
    # get the tool name and parameters from the json
    # this is an example of how the json should look like
    # tool_call_json = {'actionName': 'SaveFile', 'filename': 'test.txt', 'contents': 'Hello, World!'}

    #make sure that the actionName is in the json
    if "actionName" not in tool_call_json:
        return "No actionName provided in the tool call."

    tool_name = tool_call_json["actionName"]

    # get all of the parameters except the actionName as a dictionary
    params = {key: value for key, value in tool_call_json.items() if key != "actionName"}

    # first make sure that the tool exists
    if tool_name not in tool_dict:
        return f"Tool {tool_name} not found."

    # then make sure all of the parameters in params are in the tool's parameters
    tool_params = tool_dict[tool_name]["params"]
    if len(params) > len(tool_params):
        return f"Tool {tool_name} expects {len(tool_params)} parameters, but {len(params)} were provided."

    # then check if the parameters are named correctly
    for key, _ in params.items():
        if key not in tool_params:
            return f"Tool {tool_name} does not have a parameter named {key}."

    # if everything is correct, execute the tool (in a try-except block)
    try:
        return tool_dict[tool_name]["func"](**params)
    except Exception as e:
        return f"Error executing tool {tool_name}: {e}"


if __name__ == "__main__":
    tools = load_tools_from_file("FileTools.py")
    tools = load_tools_from_files()
    tool_query = {"actionName": "search_wikipedia", "query": "Hello, World!"}
    tool_return = execute_tool(tool_query, tools)
    print(tool_return)

