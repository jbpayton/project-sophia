from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from queue import Queue
import logging
import re
import json
from datetime import datetime
from CoreAgentStateMachine import State, ProcessContext, StateMachine, StateFlags


logger = logging.getLogger("AgentStates")


@dataclass
class ScratchMemory:
    """State machine's working memory for task processing"""
    task_stack: List[str] = field(default_factory=list)
    completed_actions: List[Dict] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    thoughts: List[str] = field(default_factory=list)

    def add_thought(self, thought: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.thoughts.append(f"[{timestamp}] {thought}")

    def add_completed_action(self, action: Dict):
        self.completed_actions.append(action)

    def clear(self):
        self.task_stack.clear()
        self.completed_actions.clear()
        self.variables.clear()
        self.thoughts.clear()


@dataclass
class AgentContext:
    """Internal: Full processing context"""
    current_task: str
    scratch_memory: ScratchMemory
    tools_handler: Any
    message_history: List[Dict[str, str]]
    llm_call: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

class ReceivedState(State[Any]):
    def process(self, context: ProcessContext[Any]) -> None:
        # Clear memory at start
        context.user_context.scratch_memory.clear()
        tools_info = context.user_context.tools_handler.get_tools_description()
        conversation = self._format_conversation_history(context.user_context.message_history[-5:])

        # Analyze if task needs tools
        prompt = f"""
        Task: {context.user_context.current_task}
        Recent conversation: {conversation}
        Available Tools: {tools_info}

        Determine if this task:
        1. Requires using any of the available tools?
        2. Is asking about previous conversation or context?
        3. Is asking a general question?

        Consider carefully:
        - Questions about numbers might need calculation even if not explicitly asked
        - Questions about previous messages need conversation context
        - General questions about capabiliiies or general knowledge need no tools

        Respond with only:
        OPERATIONAL if it needs any tool operations
        CONVERSATIONAL if it's a general question or needs context
        """

        response = context.user_context.llm_call([{"role": "user", "content": prompt}])

        # Set initial task
        context.user_context.scratch_memory.task_stack.append(context.user_context.current_task)

        if "CONVERSATIONAL" in response:
            context.user_context.scratch_memory.variables['direct_response'] = True
            context.flags.set("NEEDS_RESPONSE")
            return

        # Analyze task structure if operational
        subtasks = self._analyze_task_structure(context.user_context)
        if subtasks:
            context.user_context.scratch_memory.task_stack.clear()
            for subtask in subtasks:
                context.user_context.scratch_memory.task_stack.append(subtask)

        context.flags.set("NEEDS_THINKING")
        logger.debug(f"📋 Task Stack: {context.user_context.scratch_memory.task_stack}")

    def _format_conversation_history(self, messages: List[Dict[str, str]]) -> str:
        return "\n".join(
            f"{msg['content']}"
            for msg in messages
        )

    def _analyze_task_structure(self, context) -> List[str]:
        """Task analysis with initial check for operation need"""
        if not context.scratch_memory.task_stack[-1]:
            return []

        prompt = f"""
        Task: {context.scratch_memory.task_stack[-1]}

        Available Tools: {context.tools_handler.get_tools_description()}

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


class ThinkingState(State[Any]):
    def process(self, context: ProcessContext[Any]) -> None:
        thought = self._generate_thought(context.user_context)

        if not self._is_repetitive_thought(
                thought,
                context.user_context.scratch_memory.thoughts
        ):
            context.user_context.scratch_memory.add_thought(thought)
            context.output_queue.put({
                "content": thought,
                "sender": context.user_context.metadata.get("agent_name", "Agent"),
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            logger.info(f"💭 Agent: {thought}")

            if "RESPOND:" in thought:
                context.flags.set("NEEDS_RESPONSE")
                return

            # Check for tool usage intent
            tool_names = context.user_context.tools_handler.get_tools_list()
            intends_tool_use = any(
                re.search(rf'use.*{tool}', thought.lower())
                for tool in tool_names
            )

            if intends_tool_use:
                context.flags.set("NEEDS_TOOL")
            elif self._is_task_complete(context.user_context):
                context.flags.set("TASK_COMPLETE")
            else:
                context.flags.set("CONTINUE_THINKING")

    def _generate_thought(self, context) -> str:
        current_task = context.scratch_memory.task_stack[-1]
        actions_taken = self._format_completed_actions(context.scratch_memory.completed_actions)
        conversation = self._format_conversation_history(context.message_history[-5:])

        prompt = f"""
        Current task: {current_task}
        Recent conversation: {conversation}
        Actions completed: {actions_taken}
        Available tools: {context.tools_handler.get_tools_description()}

        IMPORTANT DECISION GUIDELINES:
        1. If this is a question about the conversation or previous context, respond directly with "RESPOND:" followed by your answer.
        2. If this requires actual calculation or file operations, specify which tool you need.
        3. If this is a general question or conversation, respond directly without using tools.

        What is your immediate next step?
        """
        return context.llm_call([{"role": "user", "content": prompt}])

    def _is_repetitive_thought(self, new_thought: str, previous_thoughts: List[str],
                               similarity_threshold: float = 0.9) -> bool:
        if not previous_thoughts:
            return False
        last_thought = previous_thoughts[-1]
        shorter, longer = sorted([new_thought, last_thought], key=len)
        similarity = sum(1 for a, b in zip(shorter, longer) if a == b) / len(longer)
        return similarity > similarity_threshold

    def _is_task_complete(self, context) -> bool:
        # [Implementation remains the same as original]
        pass

    def _format_completed_actions(self, actions: List[Dict]) -> str:
        if not actions:
            return "None"
        return "\n".join(
            f"- Used {action['tool']['actionName']}: Result = {action['result']}"
            for action in actions
        )

    def _format_conversation_history(self, messages: List[Dict[str, str]]) -> str:
        return "\n".join(
            f"{msg['content']}"
            for msg in messages
        )


class ActionState(State[Any]):
    def process(self, context: ProcessContext[Any]) -> None:
        # Generate and process tool call
        tool_response = self._generate_tool_call(context.user_context)
        result = context.user_context.tools_handler.process_tool_response(tool_response)

        if not result.success:
            logger.error(f"❌ Failed to process tool response: {result.error}")
            context.flags.set("TOOL_ERROR")
            return

        execution_results = []
        for tool_call in result.tool_calls:
            # Log tool call
            tool_call_json = json.dumps(tool_call, indent=2)
            logger.info(f"🔧 Tool Call: {tool_call_json}")
            context.output_queue.put({
                "content": tool_call_json,
                "sender": context.user_context.metadata.get("agent_name", "Agent"),
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })

            # Execute tool
            execution_result = context.user_context.tools_handler.execute_tool(tool_call)
            execution_results.append(execution_result)

            if not execution_result.success:
                logger.error(f"❌ Tool execution failed: {execution_result.error}")
                context.flags.set("TOOL_ERROR")
                return

            # Record successful execution
            context.user_context.scratch_memory.add_completed_action({
                "tool": tool_call,
                "result": execution_result.result,
                "success": execution_result.success
            })

            # Log result
            logger.info(f"📊 Tool result: {execution_result.result}")
            context.output_queue.put({
                "content": f"Tool execution result: {execution_result.result}",
                "sender": "system",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })

        # Store last execution result for reflection
        context.user_context.scratch_memory.variables['last_execution'] = execution_results[-1]
        context.flags.set("NEEDS_REFLECTION")

    def _generate_tool_call(self, context) -> str:
        prompt = f"""
        Current task: {context.scratch_memory.task_stack[-1]}
        Available tools: {context.tools_handler.get_tools_description()}

        You must generate a valid tool call in JSON format with 'actionName' and required parameters.
        Do not describe the tool call, just generate the JSON.

        The response must be in the format:
        {{"actionName": "tool_name", "param1": "value1", ...}}
        """
        return context.llm_call([{"role": "user", "content": prompt}])

class ReflectionState(State[Any]):
    def process(self, context: ProcessContext[Any]) -> None:
        last_execution = context.user_context.scratch_memory.variables.get('last_execution')
        if not last_execution:
            context.flags.set("REFLECTION_ERROR")
            logger.error("❌ No execution result found for reflection")
            return

        reflection_prompt = f"""
        Current task: {context.user_context.scratch_memory.task_stack[-1]}
        Tool used: {context.user_context.scratch_memory.completed_actions[-1]['tool']}
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

        reflection = context.user_context.llm_call([{"role": "user", "content": reflection_prompt}])

        # Log reflection
        logger.info(f"🤔 Reflection: {reflection}")
        context.output_queue.put({
            "content": reflection,
            "sender": context.user_context.metadata.get("agent_name", "Agent"),
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })

        # Check if task is complete
        task_complete = bool(re.search(r'task_complete:\s*yes', reflection.lower()))
        if task_complete:
            # Store current stack for logging
            stack_before = context.user_context.scratch_memory.task_stack.copy()
            completed_task = context.user_context.scratch_memory.task_stack.pop()

            # Log task completion
            logger.debug(f"📋 Stack before pop: {stack_before}")
            logger.debug(f"📋 Stack after pop: {context.user_context.scratch_memory.task_stack}")
            logger.debug(f"✅ Completed task: {completed_task}")

            # Set appropriate flags based on remaining tasks
            if not context.user_context.scratch_memory.task_stack:
                context.flags.set("ALL_TASKS_COMPLETE")
            else:
                context.flags.set("SUBTASK_COMPLETE")
        else:
            context.flags.set("NEEDS_MORE_STEPS")


class ResponseState(State[Any]):
    def process(self, context: ProcessContext[Any]) -> None:
        response = self._generate_response(context.user_context)

        # Log and queue the response
        logger.info(f"💬 Response: {response}")
        context.output_queue.put({
            "content": response,
            "sender": context.user_context.metadata.get("agent_name", "Agent"),
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })

        # Flag that we're done
        context.flags.set("RESPONSE_COMPLETE")

    def _generate_response(self, context) -> str:
        conversation = self._format_conversation_history(context.message_history[-5:])
        prompt = f"""
        Task: {context.current_task}
        Recent conversation: {conversation}
        Completed actions: {context.scratch_memory.completed_actions}
        Your thoughts: {context.scratch_memory.thoughts}

        Generate a clear and concise response that addresses the user's request.
        Focus on the results and avoid mentioning internal processes unless specifically asked.
        """
        return context.llm_call([{"role": "user", "content": prompt}])

    def _format_conversation_history(self, messages: List[Dict[str, str]]) -> str:
        return "\n".join(
            f"{msg['content']}"
            for msg in messages
        )

class TerminalState(State[Any]):
    """A final state that does nothing but mark completion"""
    def process(self, context: ProcessContext[Any]) -> None:
        pass

class ErrorState(State[Any]):
    def process(self, context: ProcessContext[Any]) -> None:
        error_msg = "An error occurred during processing"

        logger.error(f"❌ {error_msg}")
        context.output_queue.put({
            "content": error_msg,
            "sender": "system",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })

        context.flags.set("ERROR_HANDLED")

# create_agent_state_machine function
def create_agent_state_machine(
    tools_handler: Any,
    llm_call: Callable,
    metadata: Optional[Dict[str, Any]] = None
) -> StateMachine[str, AgentContext]:
    machine = StateMachine[str, AgentContext]()

    def create_context(task: str, message_history: List[Dict[str, str]]) -> AgentContext:
        return AgentContext(
            current_task=task,
            scratch_memory=ScratchMemory(),
            tools_handler=tools_handler,
            message_history=message_history,
            llm_call=llm_call,
            metadata=metadata or {}
        )

    original_process = machine.process
    def process_with_context(task: str, message_history: List[Dict[str, str]], output_queue: Queue) -> str:
        context = create_context(task, message_history)
        return original_process(context, output_queue)

    machine.process = process_with_context

    # Register all states
    machine.register_state("received", ReceivedState())
    machine.register_state("thinking", ThinkingState())
    machine.register_state("action", ActionState())
    machine.register_state("reflection", ReflectionState())
    machine.register_state("response", ResponseState())
    machine.register_state("terminal", TerminalState())
    machine.register_state("error", ErrorState())

    # Configure transitions

    # Received state transitions
    machine.add_transition(
        "received",
        "response",
        lambda ctx: ctx.flags.has("NEEDS_RESPONSE")
    )
    machine.add_transition(
        "received",
        "thinking",
        lambda ctx: ctx.flags.has("NEEDS_THINKING")
    )

    # Thinking state transitions
    machine.add_transition(
        "thinking",
        "response",
        lambda ctx: ctx.flags.has("NEEDS_RESPONSE") or ctx.flags.has("TASK_COMPLETE")
    )
    machine.add_transition(
        "thinking",
        "action",
        lambda ctx: ctx.flags.has("NEEDS_TOOL")
    )
    machine.add_transition(
        "thinking",
        "thinking",
        lambda ctx: ctx.flags.has("CONTINUE_THINKING")
    )

    # Action state transitions
    machine.add_transition(
        "action",
        "thinking",
        lambda ctx: ctx.flags.has("TOOL_ERROR")
    )
    machine.add_transition(
        "action",
        "reflection",
        lambda ctx: ctx.flags.has("NEEDS_REFLECTION")
    )

    # Reflection state transitions
    machine.add_transition(
        "reflection",
        "thinking",
        lambda ctx: (
                ctx.flags.has("REFLECTION_ERROR") or
                ctx.flags.has("SUBTASK_COMPLETE") or
                ctx.flags.has("NEEDS_MORE_STEPS")
        )
    )
    machine.add_transition(
        "reflection",
        "response",
        lambda ctx: ctx.flags.has("ALL_TASKS_COMPLETE")
    )

    # Response state transitions
    machine.add_transition(
        "response",
        "terminal",
        lambda ctx: ctx.flags.has("RESPONSE_COMPLETE")
    )

    # Change error state transition similarly
    machine.add_transition(
        "error",
        "terminal",
        lambda ctx: ctx.flags.has("ERROR_HANDLED")
    )

    # Global error transitions from any state to error state
    for state in ["received", "thinking", "action", "reflection", "response"]:
        machine.add_transition(
            state,
            "error",
            lambda ctx: ctx.flags.has("ERROR")
        )

    # Set the initial state
    machine.set_start_state("received")

    return machine

