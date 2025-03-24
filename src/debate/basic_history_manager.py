from src.shared.models import AgnetConfig, MessageModel

class BasicHistoryManager:
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