import asyncio

class GDModerator:
    def __init__(self, topic, keywords):
        self.topic = topic
        self.keywords = keywords
        self.user_hand_raised = False
        self.active_speaker = "Bot_A" # Initial speaker
        self.transcript_history = []

    async def run_meeting_loop(self, websocket):
        """Main loop that manages the turn-taking."""
        while True:
            if self.active_speaker == "User":
                # Wait for user to finish (detected by silence or 'Lower Hand' click)
                await asyncio.sleep(1)
            else:
                # Bot Turn: Call your run_ollama function with Persona logic
                response = await self.get_bot_turn(self.active_speaker)
                await websocket.send_json({"type": "BOT_AUDIO", "text": response})
               
                # After bot finishes, if hand is raised, switch to User
                if self.user_hand_raised:
                    self.active_speaker = "User"
                    self.user_hand_raised = False
                    await websocket.send_json({"type": "MODERATOR_SIGNAL", "msg": "YOUR_TURN"})
                else:
                    # Alternate between Bot A and Bot B
                    self.active_speaker = "Bot_B" if self.active_speaker == "Bot_A" else "Bot_A"

    async def get_bot_turn(self, persona):
        # Implementation of your run_ollama with Persona system prompt
        pass