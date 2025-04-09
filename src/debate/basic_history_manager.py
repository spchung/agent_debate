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
    
    def to_msg_array(self, agent_perspective: AgnetConfig, omit_moderator: bool = False) -> list[dict]:
        messages = self.messages.copy()
        if omit_moderator:
            messages = [msg for msg in messages if msg.agent_config.type != 'moderator']
        
        result = []
        for msg in messages:
            result.append(msg.to_dict(agent_perspective=agent_perspective))
        return result

    def get_messages_of_agent(self, agent_perspective: AgnetConfig) -> list[dict]:
        messages = self.messages.copy()
        result = []
        for msg in messages:
            if msg.agent_config.id == agent_perspective.id:
                result.append(msg.to_dict(agent_perspective=agent_perspective))
        return result

    def get_last_message(self) -> MessageModel | None:
        agent_mesages = [msg for msg in self.messages if msg.agent_config.type != 'moderator']
        if len(agent_mesages) == 0:
            return None
        return self.messages[-1]