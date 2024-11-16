from typing import List, Dict, Any, Optional, NamedTuple
from dataclasses import dataclass
import json
import re
from ToolLoader import load_tools_from_string, load_tools_from_file, get_tool_name_and_dict, execute_tool


class ToolExecutionResult(NamedTuple):
    """Result of a tool execution"""
    success: bool
    result: Any
    error: Optional[str] = None


class ToolCallParseResult(NamedTuple):
    """Result of parsing tool calls from text"""
    success: bool
    tool_calls: List[Dict[str, Any]]
    remaining_text: str
    error: Optional[str] = None


class ToolHandler:
    """Manages tool loading, execution, and related operations"""

    def __init__(self):
        self.tools: Dict[str, Dict] = {}

    def add_tool(self, tool_definition) -> bool:
        """
        Add a tool from either a function, string definition, or file.
        Returns True if tool was successfully added.
        """
        try:
            if isinstance(tool_definition, str):
                if tool_definition.endswith('.py'):
                    new_tools = load_tools_from_file(tool_definition)
                else:
                    new_tools = load_tools_from_string(tool_definition)
                self.tools.update(new_tools)
            else:
                name, tool_dict = get_tool_name_and_dict(tool_definition)
                if tool_dict:
                    self.tools[name] = tool_dict
            return True
        except Exception as e:
            print(f"Failed to add tool: {str(e)}")
            return False

    def remove_tool(self, tool_name: str) -> bool:
        """Remove a tool by name."""
        if tool_name in self.tools:
            del self.tools[tool_name]
            return True
        return False

    def has_tools(self) -> bool:
        """Check if any tools are available."""
        return bool(self.tools)

    def get_tools_list(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.tools.keys())

    def get_tools_description(self) -> str:
        """Get formatted description of all tools."""
        if not self.tools:
            return "No tools available."

        descriptions = []
        for name, tool_dict in self.tools.items():
            params_str = ", ".join(tool_dict["params"])
            descriptions.append(f"""
Tool: {name}
Description: {tool_dict["description"]}
Parameters: {params_str}
""")
        return "\n".join(descriptions)

    def process_tool_response(self, text: str) -> ToolCallParseResult:
        """
        Extract and validate tool calls from text response.
        Handles both JSON block format and inline JSON.
        """
        # Remove code block markers if present
        text = text.replace('```json', '').replace('```', '')
        text_without_json = text

        pattern = r'\{.*?\}'
        matches = re.findall(pattern, text, re.DOTALL)

        tool_calls = []
        for match in matches:
            try:
                json_obj = json.loads(match)
                if "actionName" in json_obj:
                    # Validate tool exists
                    if json_obj["actionName"] not in self.tools:
                        return ToolCallParseResult(
                            success=False,
                            tool_calls=[],
                            remaining_text=text,
                            error=f"Unknown tool: {json_obj['actionName']}"
                        )

                    # Validate parameters
                    tool_params = self.tools[json_obj["actionName"]]["params"]
                    provided_params = set(json_obj.keys()) - {"actionName"}
                    required_params = set(tool_params)

                    if not required_params.issubset(provided_params):
                        missing = required_params - provided_params
                        return ToolCallParseResult(
                            success=False,
                            tool_calls=[],
                            remaining_text=text,
                            error=f"Missing parameters for {json_obj['actionName']}: {missing}"
                        )

                    tool_calls.append(json_obj)
                    text_without_json = text_without_json.replace(match, "")
            except json.JSONDecodeError:
                continue

        if not tool_calls:
            return ToolCallParseResult(
                success=False,
                tool_calls=[],
                remaining_text=text,
                error="No valid tool calls found in response"
            )

        return ToolCallParseResult(
            success=True,
            tool_calls=tool_calls,
            remaining_text=text_without_json.strip()
        )

    def execute_tool(self, tool_call: Dict[str, Any]) -> ToolExecutionResult:
        """
        Execute a tool call and return the result.
        Handles execution errors and validation.
        """
        try:
            result, success = execute_tool(tool_call, self.tools)
            return ToolExecutionResult(
                success=success,
                result=result,
                error=None if success else f"Tool execution failed: {result}"
            )
        except Exception as e:
            return ToolExecutionResult(
                success=False,
                result=None,
                error=f"Error executing tool: {str(e)}"
            )

    def validate_tool_call(self, tool_call: Dict[str, Any]) -> Optional[str]:
        """
        Validate a tool call without executing it.
        Returns error message if invalid, None if valid.
        """
        if "actionName" not in tool_call:
            return "Tool call missing 'actionName'"

        tool_name = tool_call["actionName"]
        if tool_name not in self.tools:
            return f"Unknown tool: {tool_name}"

        tool_params = self.tools[tool_name]["params"]
        provided_params = set(tool_call.keys()) - {"actionName"}
        required_params = set(tool_params)

        if not required_params.issubset(provided_params):
            missing = required_params - provided_params
            return f"Missing parameters: {missing}"

        return None


# Example usage and tests
if __name__ == "__main__":
    # Create handler
    handler = ToolHandler()

    # Test tool definitions
    calc_tool = """
def calculate(expression: str):
    '''Safely evaluate a mathematical expression.'''
    try:
        return eval(expression)
    except Exception as e:
        return f"Error: {str(e)}"
"""

    file_tool = """
def save_file(filename: str, content: str):
    '''Save content to a file.'''
    with open(filename, 'w') as f:
        f.write(content)
    return f"File {filename} saved successfully"
"""

    # Add tools
    print("=== Adding Tools ===")
    handler.add_tool(calc_tool)
    handler.add_tool(file_tool)
    print(f"Available tools: {handler.get_tools_list()}")
    print("\nTool descriptions:")
    print(handler.get_tools_description())

    # Test tool response parsing
    print("\n=== Testing Tool Response Parsing ===")

    # Test valid response
    response1 = """
Let me calculate that for you:
```json
{"actionName": "calculate", "expression": "2 + 2"}
```
"""
    result1 = handler.process_tool_response(response1)
    print("\nValid response parsing:")
    print(f"Success: {result1.success}")
    print(f"Tool calls: {result1.tool_calls}")
    print(f"Remaining text: {result1.remaining_text}")

    # Test invalid response
    response2 = """
Let me try to save this:
```json
{"actionName": "save_file"}
```
"""
    result2 = handler.process_tool_response(response2)
    print("\nInvalid response parsing:")
    print(f"Success: {result2.success}")
    print(f"Error: {result2.error}")

    # Test tool execution
    print("\n=== Testing Tool Execution ===")

    # Valid execution
    valid_tool_call = {"actionName": "calculate", "expression": "3 * 4"}
    execution_result = handler.execute_tool(valid_tool_call)
    print("\nValid tool execution:")
    print(f"Success: {execution_result.success}")
    print(f"Result: {execution_result.result}")

    # Invalid execution
    invalid_tool_call = {"actionName": "calculate", "expression": "1/0"}
    execution_result = handler.execute_tool(invalid_tool_call)
    print("\nInvalid tool execution:")
    print(f"Success: {execution_result.success}")
    print(f"Error: {execution_result.error}")

    # Invalid execution
    invalid_tool_call = {"actionName": "quackalate", "expression": "1/0"}
    execution_result = handler.execute_tool(invalid_tool_call)
    print("\nInvalid tool execution 2:")
    print(f"Success: {execution_result.success}")
    print(f"Error: {execution_result.error}")