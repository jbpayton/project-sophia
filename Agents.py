import time
import re
from typing import List, Dict, Optional, Any, Tuple, Union

from langchain.agents.agent import ExceptionTool
from langchain.agents.agent_toolkits import FileManagementToolkit
from langchain.agents.tools import InvalidTool
from langchain.chains.summarize import load_summarize_chain

from GraphStoreMemory import GraphStoreMemory

from langchain import PromptTemplate, LLMChain
from langchain.agents import StructuredChatAgent, AgentExecutor
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage, AgentAction, AgentFinish, OutputParserException, \
    Document
from langchain.tools import Tool, DuckDuckGoSearchRun, BaseTool
from langchain.utils import get_color_mapping

from tools import paged_web_browser
from tools import image_generator_tool
from tools.ToolRegistry import ToolRegistry


class DialogueAgent:
    def __init__(
            self,
            name: str,
            system_message: SystemMessage = None,
            model=None,
            TTSEngine=None,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.TTSEngine = TTSEngine
        self.prefix = f"{self.name}: "
        self.reset()

    def reset(self):
        self.message_history = ["Here is the conversation so far."]

    def send(self) -> str:
        if self.model and self.system_message:
            print(f"{self.name}: ")
            message = self.model(
                [
                    self.system_message,
                    HumanMessage(content="\n".join(self.message_history + [self.prefix])),
                ]
            )
            if self.TTSEngine:
                self.TTSEngine.speak(message.content)

            return message.content
        else:
            raise NotImplementedError

    def receive(self, name: str, message: str) -> None:
        # create a timestamped message
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        self.message_history.append(f"({timestamp}) {name}: {message}")


class SelfModifiableAgentExecutor(AgentExecutor):
    @property
    def _chain_type(self) -> str:
        pass

    def _take_next_step(
            self,
            name_to_tool_map: Dict[str, BaseTool],
            color_mapping: Dict[str, str],
            inputs: Dict[str, str],
            intermediate_steps: List[Tuple[AgentAction, str]],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        """Take a single step in the thought-action-observation loop.

        Override this to take control of how the agent makes and acts on choices.
        """
        try:
            intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)

            # Call the LLM to see what to do.
            output = self.agent.plan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )
        except OutputParserException as e:
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                raise e
            text = str(e)
            if isinstance(self.handle_parsing_errors, bool):
                if e.send_to_llm:
                    observation = str(e.observation)
                    text = str(e.llm_output)
                else:
                    observation = "Invalid or incomplete response"
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                observation = self.handle_parsing_errors(e)
            else:
                raise ValueError("Got unexpected type of `handle_parsing_errors`")
            output = AgentAction("_Exception", observation, text)
            if run_manager:
                run_manager.on_agent_action(output, color="green")
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = ExceptionTool().run(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            return [(output, observation)]
        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            return output
        actions: List[AgentAction]
        if isinstance(output, AgentAction):
            actions = [output]
        else:
            actions = output
        result = []
        for agent_action in actions:
            if run_manager:
                run_manager.on_agent_action(agent_action, color="green")
            # Otherwise we lookup the tool
            if agent_action.tool in name_to_tool_map:
                tool = name_to_tool_map[agent_action.tool]
                return_direct = tool.return_direct
                color = color_mapping[agent_action.tool]
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                if return_direct:
                    tool_run_kwargs["llm_prefix"] = ""
                # We then call the tool on the tool input to get an observation
                observation = tool.run(
                    agent_action.tool_input,
                    verbose=self.verbose,
                    color=color,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            else:
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                observation = InvalidTool().run(
                    {
                        "requested_tool_name": agent_action.tool,
                        "available_tool_names": list(name_to_tool_map.keys()),
                    },
                    verbose=self.verbose,
                    color=None,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            result.append((agent_action, observation))
        return result

    def _call(
            self,
            inputs: Dict[str, str],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run text through and get agent response."""
        # Construct a mapping of tool name to tool for easy lookup
        num_tools = len(self.tools)
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green", "red"]
        )
        intermediate_steps: List[Tuple[AgentAction, str]] = []
        # Let's start tracking the number of iterations and time elapsed
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        # We now enter the agent loop (until it returns something).
        while self._should_continue(iterations, time_elapsed):
            if num_tools != len(ToolRegistry().get_tools(self._lc_kwargs.get("name", None))):
                # If the number of tools has changed, update the mapping
                self.tools = ToolRegistry().get_tools(self._lc_kwargs.get("name", None))
                name_to_tool_map = {tool.name: tool for tool in self.tools}
                # We construct a mapping from each tool to a color, used for logging.
                color_mapping = get_color_mapping(
                    [tool.name for tool in self.tools], excluded_colors=["green", "red"]
                )
                num_tools = len(self.tools)

            next_step_output = self._take_next_step(
                name_to_tool_map,
                color_mapping,
                inputs,
                intermediate_steps,
                run_manager=run_manager,
            )
            if isinstance(next_step_output, AgentFinish):
                return self._return(
                    next_step_output, intermediate_steps, run_manager=run_manager
                )

            intermediate_steps.extend(next_step_output)
            if len(next_step_output) == 1:
                next_step_action = next_step_output[0]
                # See if tool should return directly
                tool_return = self._get_tool_return(next_step_action)
                if tool_return is not None:
                    return self._return(
                        tool_return, intermediate_steps, run_manager=run_manager
                    )
            iterations += 1
            time_elapsed = time.time() - start_time
        output = self.agent.return_stopped_response(
            self.early_stopping_method, intermediate_steps, **inputs
        )
        return self._return(output, intermediate_steps, run_manager=run_manager)


class DialogueAgentWithTools(DialogueAgent):
    def __init__(
            self,
            name: str,
            system_message: SystemMessage,
            model,
            tools: List,
            TTSEngine=None,
    ) -> None:
        super().__init__(name, system_message, model)
        self.tools = tools
        self.TTSEngine = TTSEngine
        self.needs_to_think_more = False
        self.has_picture_to_show = False
        self.image_to_show = ""
        chat = ChatOpenAI(
            model_name='gpt-3.5-turbo-16k'
        )
        self.graph_store = GraphStoreMemory(model=chat, agent_name=name)
        self.message_history = self.graph_store.conversation_logger.load_last_n_lines(100)
        ToolRegistry().set_tools(name, self.tools)

    def receive(self, name: str, message: str) -> None:
        # create a timestamped message
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        message_to_log = f"({timestamp}) {name}: {message}"

        self.graph_store.accept_message(message_to_log)
        self.message_history.append(message_to_log)

    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """

        print("Number of messages in history: ", len(self.message_history))

        if not self.needs_to_think_more:
            conversation_mode_prompt = "You can choose to express your emotions by leading a sentence with " \
                                       "parenthesis with your emotional state. If I feel the need to use a tool, " \
                                       "I will append [search] or [recall] or [files] or [generate image] " \
                                       "this will activate a mode where deeper thought or tools can be used. " \
                                       "Prompting again will trigger the search, recall, or image generator. " \
                                       "Do not use json to activate a tool until you are in tool mode"

            print(f"{self.name}: ")
            message = self.model(
                [
                    SystemMessage(role=self.name, content=self.system_message.content + conversation_mode_prompt),
                    HumanMessage(content="\n".join(self.message_history + [self.prefix])),
                ]
            )

        else:
            self.needs_to_think_more = False
            agent_chain = SelfModifiableAgentExecutor.from_agent_and_tools(
                agent=StructuredChatAgent.from_llm_and_tools(llm=self.model,
                                                             tools=self.tools),
                tools=self.tools,
                max_iterations=10,
                verbose=True,
                memory=ConversationBufferMemory(memory_key="chat_history", input_key='input', output_key="output"),
                return_intermediate_steps=True,
                name=self.name
            )

            # summarize conversation to this point
            summary_chain = load_summarize_chain(ChatOpenAI(temperature=0,
                                                            model_name="gpt-3.5-turbo-16k"),
                                                 chain_type="stuff")

            # turn the conversation history into a list of documents
            docs = [Document(page_content=msg, metadata={"source": "local"}) for msg in self.message_history[-100:-1]]
            summary = "This is a summary of the conversation so far: " + summary_chain.run(docs)
            print(summary)
            recent_history = ["These are the last few exchanges: "] + self.message_history[-30:-1]

            response = agent_chain({"input": "\n".join([self.system_message.content] + [summary] + recent_history)})

            message = AIMessage(content=response["output"])

            # go through the intermediate steps and log them
            for step in response["intermediate_steps"]:
                step_input = step[0]
                step_output = step[1]

                if isinstance(step_input, AgentAction):
                    # if output is longer than 256 characters, write it to a file
                    if len(step_output) > 256:
                        step_output = self.graph_store.conversation_logger.log_tool_output(step_input.tool, step_output)

                    # create a timestamped message
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    string_to_log = f"({timestamp}) System: Tool:{step_input.tool} Tool Input:{step_input.tool_input} Tool Output:{step_output} "
                    self.graph_store.accept_message(string_to_log)
                    self.message_history.append(string_to_log)

        if "[" in message.content:
            bracket_contents = re.search(r'\[.*?\]', message.content).group(0)[1:-1]
            if ".png" in bracket_contents:
                self.has_picture_to_show = True
                # get the string between the brackets
                self.image_to_show = bracket_contents
            elif "search" in bracket_contents:
                self.needs_to_think_more = True
            elif "files" in bracket_contents:
                self.needs_to_think_more = True
            elif "recall" in bracket_contents:
                self.needs_to_think_more = True
            elif "generate image" in bracket_contents:
                self.needs_to_think_more = True

        if self.TTSEngine:
            spoken = message.content
            if "[" in spoken:
                spoken = re.sub(r'\[.*?\]', '', spoken)
            if self.needs_to_think_more:
                self.TTSEngine.speak_async(spoken)
            else:
                self.TTSEngine.speak(spoken)

        # create a timestamped message
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        message_to_log = f"({timestamp}) {self.name}: {message.content}"

        self.graph_store.accept_message(message_to_log)
        self.message_history.append(message_to_log)

        return message.content


class UserAgent(DialogueAgent):
    def __init__(
            self,
            name: str,
            system_message: SystemMessage = None,
            model=None,
            stt_engine=None,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.stt_engine = stt_engine
        self.prefix = f"{self.name}: "
        self.reset()

    def send(self) -> str:
        if self.stt_engine:
            message = self.stt_engine.listen()
        else:
            message = input(f"\n{self.prefix}")
        return message

    def receive(self, name: str, message: str) -> None:
        print(f"{name}: {message}")


def get_avatar_agent(profile=None, avatar_tts=None):
    # Define system prompts for our  agent
    system_prompt_avatar = SystemMessage(role=profile['name'],
                                         content=profile['personality'])

    # initialize file management tools
    file_tools = FileManagementToolkit(
        selected_tools=["read_file", "write_file", "list_directory", "copy_file", "move_file", "file_delete"]
    ).get_tools()

    tools = [DuckDuckGoSearchRun(),
             paged_web_browser,
             image_generator_tool] + file_tools

    # Initialize our agent with role and system prompt
    avatar = DialogueAgentWithTools(name=profile['name'],
                                    system_message=system_prompt_avatar,
                                    model=ChatOpenAI(model_name='gpt-4', streaming=True,
                                                     callbacks=[StreamingStdOutCallbackHandler()]),
                                    tools=tools,
                                    TTSEngine=avatar_tts)
    return avatar
