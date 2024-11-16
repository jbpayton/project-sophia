from typing import List, Dict, Any, Optional
from queue import Queue
from GeneralAgentStateMachine import create_agent_state_machine
from ToolHandler import ToolHandler
from LLMProviders import LLMConfig, get_chat_llm

# Agent.py - Add at the top with imports
import logging


def setup_logging(verbose: bool = False):
    """Configure logging with proper filters"""
    # Set base logging level
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        datefmt='%H:%M:%S',
        level=logging.WARNING  # Base level for all loggers
    )

    # Get our specific loggers
    state_machine_logger = logging.getLogger("StateMachine")
    agent_states_logger = logging.getLogger("AgentStates")

    # Set levels for our loggers only
    level = logging.DEBUG if verbose else logging.INFO
    state_machine_logger.setLevel(level)
    agent_states_logger.setLevel(level)

    # Explicitly set other loggers to WARNING or higher
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

class Agent:
    """
    Agent class that integrates StateMachine and ToolHandler for cleaner operation.
    Focuses on configuration and coordination rather than implementation details.
    """

    def __init__(self,
                 name: str,
                 personality: Dict[str, Any],
                 llm_config: LLMConfig,
                 verbose: bool = False,
                 message_handler: Optional[Any] = None):

        # Set up logging first
        setup_logging(verbose)

        # Core components
        self.name = name
        self.personality = personality
        self.llm_config = llm_config

        # Initialize LLM
        self.llm = get_chat_llm(llm_config)

        # State and memory management
        self.messages: List[Dict[str, str]] = []
        self.output_queue = Queue()

        # Tool and state management
        self.tools = ToolHandler()
        self.state_machine = create_agent_state_machine(
            tools_handler=self.tools,
            llm_call=self.llm.generate_response,
            metadata={"agent_name": self.name}
        )

        self._update_system_prompt()

    def _build_system_prompt(self) -> str:
        # Simpler prompt that preserves user's specified personality
        personality_str = "\n".join(f"- {key}: {value}"
                                    for key, value in self.personality.items())

        return f"""You are {self.name}.

        Personality:
        {personality_str}
    
        Available tools:
        {self.tools.get_tools_list()}"""


    def _format_system_message(self, role: str, content: str) -> Dict[str, str]:
        """Format a message using the current message format configuration"""
        return self.llm_config.message_format.format_system_message(
            role=role,
            content=content
        )

    def _update_system_prompt(self):
        """Updates the system prompt based on current agent state"""
        system_prompt = self._build_system_prompt()
        formatted_prompt = self._format_system_message("system", system_prompt)

        if not self.messages:
            self.messages.append(formatted_prompt)
        elif self.messages[0]["role"] == formatted_prompt["role"]:
            self.messages[0] = formatted_prompt
        else:
            self.messages.insert(0, formatted_prompt)

    def add_tool(self, tool_definition) -> bool:
        """Add a tool to the agent"""
        success = self.tools.add_tool(tool_definition)
        if success:
            self._update_system_prompt()
        return success

    def send(self, message: str, sender: Optional[str] = None) -> str:
        """Send a message to the agent for processing"""
        if not self.llm:
            raise ValueError("LLM not configured. Please configure LLM before sending messages.")

        # Store original conversation history
        original_history = self.messages.copy()

        # Add user message to working history
        # this is the user message, so use the sender name if available
        name = sender if sender else "User"
        # add the sender name to the message
        prefix = f"{name}: "
        # add the message to the history
        self.messages.append({"role": "user", "content": prefix + message})


        # Process through state machine
        try:
            final_state = self.state_machine.process(
                task=message,
                message_history=self.messages,
                output_queue=self.output_queue
            )

            # Restore original history and append only the input and final response
            self.messages = original_history
            self.messages.append({"role": "user", "content": prefix + message})

            # Look for just the final response in queue
            final_response = None
            while not self.output_queue.empty():
                final_response = self.output_queue.get()

            # Only update conversation history with the input and final response
            if final_response:
                # Add the original message and response to history
                self.messages = original_history
                self.messages.append({"role": "user", "content": prefix + message})

                response = f"{self.name}: {final_response['content']}"
                self.messages.append({"role": "assistant", "content": response})

                # Put only the final response back in queue for the user
                self.output_queue.put({"role": "assistant", "content": response})

        except Exception as e:
            self.messages = original_history
            raise

    def get_next_output(self) -> Optional[Dict[str, Any]]:
        """Get the next message from the output queue"""
        if not self.output_queue.empty():
            return self.output_queue.get()
        return None


if __name__ == "__main__":
    # Test tools
    calculator_tool = """
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

    # Create and configure agent
    # Create LLM configuration
    from util import load_secrets
    from LLMProviders import MessageFormat
    load_secrets("../secrets.json")

    llm_config = LLMConfig(
        model="gpt-4",  # default, could be omitted
        temperature=0.7,
        message_format=MessageFormat(style="llama"),
        api_key="sk-111111111111111111111111111111111111111111111111",
        api_base_url="http://192.168.2.94:5000/v1",
        provider = "openai"
    )

    '''
    llm_config = LLMConfig(
        model="open-mixtral-8x22b",
        temperature=0.7,
        message_format=MessageFormat(style="llama"),
        api_key=os.getenv("MISTRAL_API_KEY"),
        provider="mistral",
        rate_limit_delay=2
    )
    '''
    # Create agent
    agent = Agent(
        name="MathBot",
        personality={
            "role": "mathematics expert",
            "expertise": ["arithmetic", "algebra"],
            "style": "patient and educational"
        },
        llm_config=llm_config,
        verbose=True
    )

    # Add tools
    agent.add_tool(calculator_tool)
    agent.add_tool(file_tool)

    # Send some test messages
    print("\n=== Testing Basic Calculation ===")
    agent.send("What's 25 * 32?")

    # get queued message
    print(agent.get_next_output())

    print("\n=== Testing Multi-step Task ===")
    agent.send("Calculate 15 * 7 and 22 * 3, then save the results to 'math_results.txt'")
    print(agent.get_next_output())

    print("\n=== Testing Multi-user Interaction ===")
    agent.send("Can you help me with some math?", sender="Alice")
    print(agent.get_next_output())

    agent.send("What was Alice asking about?", sender="Bob")
    print(agent.get_next_output())