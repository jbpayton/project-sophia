from enum import Enum
from typing import Optional, Callable, Dict, Any, List, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
from queue import Queue
import json
import re


class AgentState(Enum):
    IDLE = "idle"
    RECEIVED = "received"
    THINKING = "thinking"
    ACTION = "action"
    REFLECTION = "reflection"  # New state for post-tool evaluation
    RESPONSE = "response"
    ERROR = "error"

    def __str__(self):
        return self.value


class ExecutionResult(NamedTuple):
    """Structured result from tool execution"""
    success: bool
    result: Any
    error: Optional[str] = None


@dataclass
class StateContext:
    """Context for state transitions and processing"""
    current_task: str
    scratch_memory: Any  # Will hold the ScratchMemory instance
    tools_handler: Any  # Will hold the ToolHandler instance
    message_history: List[Dict[str, str]]
    output_queue: Queue
    llm_call: Callable
    metadata: Dict[str, Any] = field(default_factory=dict)


class DefaultMessageHandler:
    """Default handler for verbose output messages with emoji indicators"""

    EMOJI_MAP = {
        "internal_thought": "💭",
        "tool_call": "🔧",
        "tool_result": "📊",
        "tool_reflection": "🤔",  # Specific emoji for reflection state
        "response": "💬",
        "error": "❌",
        "debug": "🔍",
        "state_change": "📎"
    }

    def __init__(self):
        self.logger = logging.getLogger("AgentStateMachine")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)

    def handle_message(self, message: Dict[str, Any]):
        """Process and display a message with appropriate formatting"""
        msg_type = message.get("msg_type", "")
        sender = message.get("sender", "System")
        content = message.get("content", "")
        emoji = self.EMOJI_MAP.get(msg_type, "")

        formatted_message = f"{emoji} {sender}: {content}" if emoji else f"{sender}: {content}"

        if msg_type == "error":
            self.logger.error(formatted_message)
        elif msg_type == "debug":
            self.logger.debug(formatted_message)
        else:
            self.logger.info(formatted_message)


class StateMachine:
    """Enhanced state machine with dedicated reflection state"""

    def __init__(self, verbose: bool = False, message_handler: Optional[Any] = None):
        self.state = AgentState.IDLE
        self.verbose = verbose
        self.message_handler = message_handler if message_handler else DefaultMessageHandler()
        self.max_iterations = 99
        self.current_iterations = 0

    def _log_message(self, message: Dict[str, Any]):
        if self.verbose:
            self.message_handler.handle_message(message)

    def _queue_and_log(self, context: StateContext, message: Dict[str, Any]):
        context.output_queue.put(message)
        self._log_message(message)

    def process(self, context: StateContext) -> AgentState:
        """Main processing loop with iteration limiting"""
        self.state = AgentState.RECEIVED
        self.current_iterations = 0

        try:
            while self.current_iterations < self.max_iterations:
                self._log_message({
                    "msg_type": "state_change",
                    "content": f"Current State: {self.state}",
                    "sender": "StateMachine"
                })

                self._log_task_state(context)

                self.current_iterations += 1

                try:
                    if self.state == AgentState.RECEIVED:
                        self._handle_received_state(context)
                    elif self.state == AgentState.THINKING:
                        self._handle_thinking_state(context)
                    elif self.state == AgentState.ACTION:
                        self._handle_action_state(context)
                    elif self.state == AgentState.REFLECTION:
                        if self._handle_reflection_state(context):
                            self.state = AgentState.RESPONSE
                        else:
                            self.state = AgentState.THINKING
                    elif self.state == AgentState.RESPONSE:
                        if self._handle_response_state(context):
                            break
                    elif self.state == AgentState.ERROR:
                        self._handle_error_state(context)
                        break
                except Exception as e:
                    self._handle_state_error(context, e)
                    break

            if self.current_iterations >= self.max_iterations:
                self._handle_max_iterations_reached(context)

        except Exception as e:
            self._handle_process_error(context, e)

        return self.state

    def _handle_received_state(self, context: StateContext):
        """Enhanced initial state handling with dynamic tool detection"""
        context.scratch_memory.clear()

        # Get available tools for detection
        tools_info = context.tools_handler.get_tools_description()

        prompt = f"""
    Task: {context.current_task}

    Available Tools:
    {tools_info}

    Determine if this task:
    1. Requires using any of the available tools?
    2. Is asking for specific operations using these tools?
    3. Is just asking a general question or about context?

    Respond with only:
    OPERATIONAL if it needs any tool operations
    CONVERSATIONAL if it's just a general question or context query
    """
        response = context.llm_call([{"role": "user", "content": prompt}])

        # Get list of tool names for verification
        tool_names = context.tools_handler.get_tools_list()
        has_tool_mention = any(tool.lower() in context.current_task.lower() for tool in tool_names)

        if "CONVERSATIONAL" in response and not has_tool_mention:
            context.scratch_memory.task_stack.append(context.current_task)
            context.scratch_memory.variables['direct_response'] = True
            self.state = AgentState.RESPONSE
            return

        # Otherwise proceed with normal task handling
        context.scratch_memory.task_stack.append(context.current_task)
        subtasks = self._analyze_task_structure(context)

        if subtasks:
            context.scratch_memory.task_stack.clear()
            for subtask in subtasks:
                context.scratch_memory.task_stack.append(subtask)

        self._log_task_state(context)
        self.state = AgentState.THINKING

    def _handle_thinking_state(self, context: StateContext):
        """Generate thought and determine next state"""
        thought = self._generate_thought(context)

        if not self._is_repetitive_thought(thought, context.scratch_memory.thoughts):
            context.scratch_memory.add_thought(thought)
            self._queue_and_log(context, {
                "content": thought,
                "msg_type": "internal_thought",
                "sender": context.metadata.get("agent_name", "Agent"),
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })

            # Direct response check
            if "RESPOND:" in thought:
                self.state = AgentState.RESPONSE
                return

            # Tool use check
            tool_names = context.tools_handler.get_tools_list()
            intends_tool_use = any(re.search(rf'use.*{tool}', thought.lower())
                                   for tool in tool_names)

            if intends_tool_use:
                self.state = AgentState.ACTION
            else:
                # If no tool use intended and no direct response,
                # check if we should move to response based on completion
                if self._is_task_complete(context):
                    self.state = AgentState.RESPONSE
                else:
                    self.state = AgentState.THINKING

    def _handle_action_state(self, context: StateContext):
        """Handle tool execution and transition to reflection"""
        tool_response = self._generate_tool_call(context)
        result = context.tools_handler.process_tool_response(tool_response)

        if not result.success:
            self._log_message({
                "msg_type": "error",
                "content": f"Failed to process tool response: {result.error}",
                "sender": "StateMachine"
            })
            self.state = AgentState.THINKING
            return

        execution_results = []
        for tool_call in result.tool_calls:
            self._queue_and_log(context, {
                "content": json.dumps(tool_call, indent=2),
                "msg_type": "tool_call",
                "sender": context.metadata.get("agent_name", "Agent"),
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })

            execution_result = context.tools_handler.execute_tool(tool_call)
            execution_results.append(execution_result)

            if not execution_result.success:
                self._log_message({
                    "msg_type": "error",
                    "content": f"Tool execution failed: {execution_result.error}",
                    "sender": "StateMachine"
                })
                self.state = AgentState.THINKING
                return

            context.scratch_memory.add_completed_action({
                "tool": tool_call,
                "result": execution_result.result,
                "success": execution_result.success
            })

            self._queue_and_log(context, {
                "content": f"Tool execution result: {execution_result.result}",
                "msg_type": "tool_result",
                "sender": "system",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })

        # Move to reflection state instead of thinking
        self.state = AgentState.REFLECTION
        # Store the last execution result for reflection
        context.scratch_memory.variables['last_execution'] = execution_results[-1]

    def _handle_reflection_state(self, context: StateContext) -> bool:
        """
        Handle post-tool execution reflection and completion determination.
        Returns True if task is complete, False if more processing needed.
        """
        last_execution = context.scratch_memory.variables.get('last_execution')
        if not last_execution:
            return False

        reflection_prompt = f"""
Current task: {context.scratch_memory.task_stack[-1]}
Tool used: {context.scratch_memory.completed_actions[-1]['tool']}
Result: {last_execution.result}

Evaluate the following:
1. Was the tool execution successful? (yes/no)
2. Does this result complete the current task? (yes/no)
3. Are additional steps needed? (yes/no)

Answer in the format:
EXECUTION_SUCCESSFUL: yes/no
TASK_COMPLETE: yes/no
NEEDS_MORE_STEPS: yes/no
REASON: [brief explanation]
"""
        reflection = context.llm_call([{"role": "user", "content": reflection_prompt}])

        self._queue_and_log(context, {
            "content": reflection,
            "msg_type": "tool_reflection",
            "sender": context.metadata.get("agent_name", "Agent"),
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })

        task_complete = bool(re.search(r'task_complete:\s*yes', reflection.lower()))
        if task_complete:
            stack_before = context.scratch_memory.task_stack.copy()
            completed_task = context.scratch_memory.task_stack.pop()
            self._log_message({
                "msg_type": "debug",
                "content": f"Stack before pop: {stack_before}\nStack after pop: {context.scratch_memory.task_stack}",
                "sender": "StateMachine"
            })
            self._log_message({
                "msg_type": "debug",
                "content": f"Completed task: {completed_task}",
                "sender": "StateMachine"
            })
            self._log_task_state(context)

            # If no more tasks, go to response
            if not context.scratch_memory.task_stack:
                return True
            # If more tasks, go back to thinking
            return False

        return False

    def _handle_response_state(self, context: StateContext) -> bool:
        """Generate final response and complete processing"""
        response = self._generate_response(context)

        self._queue_and_log(context, {
            "content": response,
            "msg_type": "response",
            "sender": context.metadata.get("agent_name", "Agent"),
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })

        self.state = AgentState.IDLE
        return True

    # [Error handling methods remain the same]
    def _handle_error_state(self, context: StateContext):
        """Process error state with detailed error reporting"""
        error_msg = "An error occurred during processing"
        self._queue_and_log(context, {
            "content": error_msg,
            "msg_type": "error",
            "sender": "system",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })

    def _handle_state_error(self, context: StateContext, error: Exception):
        error_msg = f"Error in state {self.state}: {str(error)}"
        self._log_message({
            "msg_type": "error",
            "content": error_msg,
            "sender": "StateMachine"
        })
        self.state = AgentState.ERROR

    def _handle_process_error(self, context: StateContext, error: Exception):
        error_msg = f"Process error: {str(error)}"
        self._log_message({
            "msg_type": "error",
            "content": error_msg,
            "sender": "StateMachine"
        })
        self.state = AgentState.ERROR

    def _handle_max_iterations_reached(self, context: StateContext):
        error_msg = "Maximum iterations reached, stopping processing"
        self._queue_and_log(context, {
            "content": error_msg,
            "msg_type": "error",
            "sender": "system",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        self.state = AgentState.ERROR

    # [Prompt building methods]
    def _build_thought_prompt(self, context: StateContext) -> str:
        current_task = context.scratch_memory.task_stack[-1]
        actions_taken = "None"
        if context.scratch_memory.completed_actions:
            actions_taken = "\n".join(
                f"- Used {action['tool']['actionName']}: Result = {action['result']}"
                for action in context.scratch_memory.completed_actions
            )

        conversation = "\n".join(
            f"{msg['role']}: {msg['content']}"
            for msg in context.message_history[-5:]  # Increased context window
        )

        return f"""
    Current task: {current_task}

    Recent conversation:
    {conversation}

    Actions completed:
    {actions_taken}

    Available tools: {context.tools_handler.get_tools_description()}

    IMPORTANT DECISION GUIDELINES:
    1. If this is a question about the conversation or previous context, you should respond directly with "RESPOND:" followed by your answer.
    2. If this requires actual calculation or file operations, specify which tool you need.
    3. If this is a general question or conversation, respond directly without using tools.

    Consider:
    1. Does this task REALLY need a tool? Only use tools for actual calculations or file operations.
    2. Is this just asking about previous conversation? Use "RESPOND:" for these cases.
    3. Is this general conversation? Use "RESPOND:" for these cases.

    What is your immediate next step?
    """

    def _build_tool_prompt(self, context: StateContext) -> str:
        return f"""
Current task: {context.scratch_memory.task_stack[-1]}
Available tools: {context.tools_handler.get_tools_description()}

You must generate a valid tool call in JSON format with 'actionName' and required parameters.
Do not describe the tool call, just generate the JSON.

The response must be in the format:
{{"actionName": "tool_name", "param1": "value1", ...}}
"""

    def _build_response_prompt(self, context: StateContext) -> str:
        return f"""
Task: {context.current_task}
Completed actions: {context.scratch_memory.completed_actions}
Your thoughts: {context.scratch_memory.thoughts}

Generate a clear and concise response that addresses the user's request.
Focus on the results and avoid mentioning internal processes unless specifically asked.
"""

    def _analyze_task_structure(self, context: StateContext) -> List[str]:
        """Task analysis with initial check for operation need"""
        if not context.scratch_memory.task_stack[-1]:
            return []

        # Quick check if this needs tool operations at all
        quick_check = f"""
    Task: {context.scratch_memory.task_stack[-1]}

    Is this a task that requires specific tool operations (like calculations or file operations), 
    or is it just asking a general question/for help?

    Respond with only one word: OPERATIONAL or CONVERSATIONAL
    """
        response = context.llm_call([{"role": "user", "content": quick_check}])

        if "CONVERSATIONAL" in response:
            context.scratch_memory.variables['direct_response'] = True
            return []

        # If we get here, it needs tools, so do full breakdown
        prompt = f"""
    Task: {context.scratch_memory.task_stack[-1]}

    Available Tools:
    {context.tools_handler.get_tools_description()}

    Break this task into minimal required tool operations.
    Each subtask must directly use one of the available tools.
    List each required tool operation as a separate step.

    SUBTASKS:
    """
        response = context.llm_call([{"role": "user", "content": prompt}])

        subtasks = []
        in_subtasks = False
        for line in response.split('\n'):
            line = line.strip()
            if line.upper() == 'SUBTASKS:':
                in_subtasks = True
                continue
            if in_subtasks and line and line[0].isdigit():
                task = re.sub(r'^\d+\.\s*', '', line).strip()
                if task:
                    subtasks.append(task)

        return subtasks

    def _is_repetitive_thought(self, new_thought: str, previous_thoughts: List[str],
                               similarity_threshold: float = 0.9) -> bool:
        """Detect repetitive thoughts"""
        if not previous_thoughts:
            return False

        last_thought = previous_thoughts[-1]
        shorter, longer = sorted([new_thought, last_thought], key=len)
        similarity = sum(1 for a, b in zip(shorter, longer) if a == b) / len(longer)
        return similarity > similarity_threshold

    def _format_task_stack(self, tasks: List[str]) -> str:
        """Format task stack for prompt clarity"""
        if not tasks:
            return "No tasks in stack."

        formatted = []
        for idx, task in enumerate(tasks):
            prefix = "Main task" if idx == 0 else f"Subtask {idx}"
            formatted.append(f"{prefix}: {task}")

        return "\n".join(formatted)

    def _format_conversation_history(self, messages: List[Dict[str, str]]) -> str:
        """Format recent conversation messages for context"""
        if not messages:
            return "No recent messages."

        formatted = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")

        return "\n".join(formatted)

    def _format_completed_actions(self, actions: List[Dict]) -> str:
        """Format completed actions for prompt clarity"""
        if not actions:
            return "No actions taken yet."

        formatted = []
        for action in actions:
            formatted.append(f"- Used {action['tool']['actionName']} tool")
            formatted.append(f"  Result: {action['result']}")
            formatted.append(f"  Success: {action['success']}")
        return "\n".join(formatted)

    def _format_recent_thoughts(self, thoughts: List[str], limit: int = 3) -> str:
        """Format recent thoughts for prompt clarity"""
        if not thoughts:
            return "No previous thoughts."
        return "\n".join(f"- {thought}" for thought in thoughts[-limit:])

    def _get_state_summary(self, context: StateContext) -> str:
        """Generate a summary of current state for debugging"""
        return f"""
Current State: {self.state}
Task: {context.scratch_memory.task_stack[-1] if context.scratch_memory.task_stack else 'No task'}
Thoughts: {len(context.scratch_memory.thoughts)}
Actions: {len(context.scratch_memory.completed_actions)}
Iterations: {self.current_iterations}/{self.max_iterations}
"""

    def reset(self):
        """Reset the state machine to initial state"""
        self.state = AgentState.IDLE
        self.current_iterations = 0

    def _generate_thought(self, context: StateContext) -> str:
        """Generate next thought with full context awareness"""
        prompt = self._build_thought_prompt(context)
        return context.llm_call([{"role": "user", "content": prompt}])

    def _is_task_complete(self, context: StateContext) -> bool:
        """Enhanced completion check with subtask awareness"""
        current_task = context.scratch_memory.task_stack[-1]
        all_tasks = context.scratch_memory.task_stack

        prompt = f"""
    Current subtask: {current_task}
    Overall task: {all_tasks[0]}

    All tasks in stack:
    {self._format_task_stack(all_tasks)}

    Recent conversation context:
    {self._format_conversation_history(context.message_history[-3:])}

    Tools used and results:
    {self._format_completed_actions(context.scratch_memory.completed_actions)}

    Your recent thoughts:
    {self._format_recent_thoughts(context.scratch_memory.thoughts)}

    Consider ONLY the current subtask:
    1. Has this specific subtask been completed? (yes/no)
    2. Were all necessary operations for this subtask performed? (yes/no)
    3. Were all operations successful? (yes/no)
    4. Are there any remaining steps for this subtask? (yes/no)

    Answer with ONLY yes/no for each question, then a brief explanation why:
    SUBTASK_COMPLETE:
    OPERATIONS_DONE:
    ALL_SUCCESSFUL:
    STEPS_REMAINING:
    REASON:
    """
        response = context.llm_call([{"role": "user", "content": prompt}])

        # Log the completion check reasoning
        self._log_message({
            "msg_type": "debug",
            "content": f"Subtask completion check:\n{response}",
            "sender": "StateMachine"
        })

        is_complete = bool(re.search(r'subtask_complete:\s*yes', response.lower()))

        if is_complete:
            # If subtask is complete, pop it from stack
            completed_task = context.scratch_memory.task_stack.pop()
            self._log_message({
                "msg_type": "debug",
                "content": f"Completed subtask: {completed_task}",
                "sender": "StateMachine"
            })

            # If stack is empty, we're done
            if not context.scratch_memory.task_stack:
                return True

            # Otherwise, continue with next subtask
            return False

        return False

    def _generate_tool_call(self, context: StateContext) -> str:
        """Generate appropriate tool call based on context"""
        prompt = self._build_tool_prompt(context)
        return context.llm_call([{"role": "user", "content": prompt}])

    def _generate_response(self, context: StateContext) -> str:
        """Generate final response considering all context"""
        prompt = self._build_response_prompt(context)
        return context.llm_call([{"role": "user", "content": prompt}])

    def _log_task_state(self, context: StateContext):
        self._log_message({
            "msg_type": "debug",
            "content": f"""
    Task Stack: {context.scratch_memory.task_stack}
    Completed Actions: {len(context.scratch_memory.completed_actions)}
    Variables: {context.scratch_memory.variables}
    """,
            "sender": "TaskState"
        })
