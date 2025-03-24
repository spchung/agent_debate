from src.shared.models import AgnetConfig, MessageModel

class DebateChatHistoryManager:
    def __init__(self):
        self.history = []
        self.moderator_id = None
        self.agents = {} # id - name
        self.messages = []  # for List[Message]
        self.raw_messages_list = []

    def register_agent_moderator(self, moderator_config: AgnetConfig):
        if self.moderator_id:
            raise ValueError("Moderator is already registered")
        
        self.moderator_id = moderator_config.id
        self.agents[moderator_config.id] = moderator_config.name
    
    def register_agent_debator(self, debator_config: AgnetConfig):
        if debator_config.id in self.agents:
            raise ValueError("Agent is already registered")
        
        self.agents[debator_config.id] = debator_config.name
    
    def add_message(self, agent_config: AgnetConfig, message: str):
        new_msg = MessageModel(agent_config=agent_config, message=message)
        
        self.messages.append(new_msg)

        # update raw representation
        self.raw_messages_list.append(new_msg.to_dict())
    
    def to_msg_array(self, agent_perspective: AgnetConfig):
        messages = []
        for msg in self.messages:
            messages.append(msg.to_dict(agent_perspective=agent_perspective))
        return messages

    ## temp fabricate memory
    def __fabricate_memory(self):
        try: 
            msgs = [
                {"agent_id": "mod", "content": "Today's topic is: 'Should mobile phones be banned in school'. Let's start with the opening statement from the proposition."},
                {"agent_id": "agent_1", "content": "Mobile phones should be banned in schools because they are a distraction to students and can be used for cheating."},
                {"agent_id": "agent_2", "content": "Mobile phones should not be banned in schools because they are a useful tool for learning and communication."},
                {"agent_id": "agent_1", "content": "Mobile phones are a distraction to students and can lead to academic dishonesty. Students often use their phones to text, browse social media, or play games during class, which can disrupt the learning environment."},
                {"agent_id": "agent_2", "content": "While it's true that mobile phones can be a distraction, they also have educational benefits. Students can use their phones to access educational apps, research information, and communicate with teachers and classmates."},
            ]

            for msg in msgs:
                agent_id = msg["agent_id"]
                agent_name = self.agents[agent_id]
                config = AgnetConfig(id=agent_id, name=agent_name)
                self.add_message(agent_config=config, message=msg["content"])
        except Exception as e:
            print(e)
