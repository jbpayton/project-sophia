import time
import re
from typing import List, Dict, Optional, Any, Tuple
from GraphStoreMemory import GraphStoreMemory

from langchain import PromptTemplate, LLMChain
from langchain.agents import StructuredChatAgent, AgentExecutor
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage, AgentAction, AgentFinish
from langchain.tools import Tool, DuckDuckGoSearchRun
from langchain.utils import get_color_mapping

from tools import paged_web_browser
from tools.ToolRegistry import ToolRegistry


class DialogueAgent:
    def __init__(
        self,
        name: str,
        system_message: SystemMessage = None,
        model = None,
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
            TTSEngine = None,
    ) -> None:
        super().__init__(name, system_message, model)
        self.tools = tools
        self.TTSEngine = TTSEngine
        self.needs_to_think_more = False
        chat = ChatOpenAI(
            model_name='gpt-3.5-turbo-16k'
        )
        self.graph_store = GraphStoreMemory(model=chat)
        ToolRegistry().set_tools(name, self.tools)

    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """

        if not self.needs_to_think_more:
            ask_for_tools_prompt = "If I feel the need to look something up, whether from the web or from your own " \
                                   "memory, I will append [search] or [recall] and will then tell you I am either " \
                                   "thinking or searching for something. I will say a lot and stall for time while I " \
                                     "am thinking. If I am searching, I will tell you what I am searching for."

            print(f"{self.name}: ")
            message = self.model(
                [
                    SystemMessage(role=self.name, content=self.system_message.content + ask_for_tools_prompt),
                    HumanMessage(content="\n".join(self.message_history + [self.prefix])),
                ]
            )

        else:
            agent_chain = SelfModifiableAgentExecutor.from_agent_and_tools(
                agent=StructuredChatAgent.from_llm_and_tools(llm=self.model,
                                                             tools=self.tools),
                tools=self.tools,
                max_iterations=10,
                verbose=True,
                memory=ConversationBufferMemory(
                    memory_key="chat_history", return_messages=True
                ),
                name=self.name
            )

            message = AIMessage(
                content=agent_chain.run(
                    input="\n".join(
                        [self.system_message.content] + self.message_history + [self.prefix]
                    )
                )
            )

        if "[" in message.content:
            self.needs_to_think_more = True
        else:
            self.needs_to_think_more = False

        if self.TTSEngine:
            spoken = message.content
            if "[" in spoken:
                spoken = re.sub(r'\[.*?\]', '', spoken)
            if self.needs_to_think_more:
                self.TTSEngine.speak_async(spoken)
            else:
                self.TTSEngine.speak(spoken)

        self.message_history.append(f"{self.name}: {message.content}")
        return message.content

class UserAgent(DialogueAgent):
    def __init__(
        self,
        name: str,
        system_message: SystemMessage = None,
        model = None,
        stt_engine = None,
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
    tools = [DuckDuckGoSearchRun(),
             paged_web_browser]

    # Initialize our agent with role and system prompt
    avatar = DialogueAgentWithTools(name=profile['name'],
                                    system_message=system_prompt_avatar,
                                    model=ChatOpenAI(model_name='gpt-4', streaming=True,
                                                     callbacks=[StreamingStdOutCallbackHandler()]),
                                    tools=tools,
                                    TTSEngine=avatar_tts)
    return avatar
