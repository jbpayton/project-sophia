from Iridescent import Agent, Community, LLMs, SearchTools, IOTools, ProgrammingTools, MetaTools, SamplePersonalities, Outputs

####
#this is a fiddly workflow, there will be others with good defaults
####

#specify a community and community tools
my_community = Community(agents=[], tools=[SearchTools.duckduckGoSearch])
my_community.set_behavior(Community.ROUND_ROBIN)

# this a system prompt for a whole community, the community members are made aware of this.
my_community.set_charter("To help answer questions from the user.")

def my_custom_tool(inputString):
    return "This tool gives you no information"

my_custom_tool_2 = "def my_custom_tool(inputString):\n    return \"This tool gives you no information\""

bob = Agent("Bob", SamplePersonalities.SoftwareEngineer)

# add the LLM, maybe here we accomodate langchain as well as regular APIs
bob.add_llm(LLMs.OpenAI.ChatGPT())

#demonstrate the usage of addtool, by the way, this could work anywhere, any time
bob.add_tool(ProgrammingTools.REPL)
bob.add_tool(my_custom_tool_2)

#This is the observability that langchain doesnt give me...
current_agent_prompt = bob.get_current_agent_prompt()

# this gives the agent exactly what you think it does...
bob.add_tool(MetaTools.AddTool)

#this is just to demonstrate serialization
bob_description_json = bob.serialize()

my_community.add_agent(bob)
# human handler is a class that can be used to handle human input/output
my_community.add_agent(Agent.get_human_proxy(human_handler))
#this adds a default generic agent
my_community.add_agent()

#This will show us how the community prompt has propagated
current_agent_prompt = bob.get_current_agent_prompt()

# defaults to standard out
my_community.set_verbose_output()

my_community.send("Can you make me a tool to check for the color in a graphics file?")

#this runs blocking (send async would send this as an async event)

my_community_2 = Community(agents=[], tools=[IOTools.file_writer, IOTools.file_reader])

# ambassador exchange information between communities, there are other community to community patterns as well
john = Agent("John", SamplePersonalities.Ambassador)

john.join_communities([my_community, my_community_2])

# we could also use community presets, from JSON or from other places
news_community = Community.SummarizationCommunity(members=3)
news_community.add_agemt(john)
my_community.set_output()

#this runs blocking (send async would send this as an async event)
my_community_2 = Community(agents=[], tools=[IOTools.file_writer, IOTools.file_reader])

# Messenger exchange information between communities, there are other community to community patterns as well
john = Agent("John", SamplePersonalities.Messenger)

john.join_communities([my_community, my_community_2])

std_yellow = Outputs.ConsoleOut(color="yellow")

# we could also use community presets, from JSON or from other places
news_community = Community.SummarizationCommunity(members=3)
news_community.add_agent(john)
my_community.set_output(std_yellow)

ragna = Agent.FromJson("CommunityBuilderAgent")
ragna.send("Create a community to make webpapges bassed on imcoming reports.")

# so, as this might suggest, all communities are in a global community registry, also communities can contain other communities. Allowing for "has-a" relationships, also serializing a whole community might be deep by default.
Community.get_global_community().serialize("global.json")

#... so we might want to be able to do this...
Community.get_global_community().serialize("global.json", structure_only=True)