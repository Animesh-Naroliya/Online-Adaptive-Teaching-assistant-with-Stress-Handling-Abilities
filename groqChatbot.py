import os
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq

# Load environment variables (needed to ensure GROQ_API_KEY is available)
load_dotenv() 

MAX_HISTORY_MESSAGES = 10 

class LLM_Chatbot:
    def __init__(self):
        # Initialize Groq LLM 
        self.llm = ChatGroq(model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=0.7)
        self.chain = self._build_chain()
        # In-memory history store, keyed by conversation_id
        self.history_store: Dict[str, ChatMessageHistory] = {}
        
        if not os.environ.get("GROQ_API_KEY"):
            print("WARNING: GROQ_API_KEY not found. Using generic fallback.")

    def _generate_system_prompt(self, user_data: Dict[str, Any]) -> str:
        """Generates the personalized, emotion-aware system prompt, using both voice and facial input."""
        
        # NOTE: app.py must send these two distinct keys.
        voice_emotion = user_data.get('voice_emotion', 'Neutral')
        facial_emotion = user_data.get('facial_emotion', 'Neutral')
        context = user_data.get('context', 'a student')
        likes = user_data.get('likes', 'learning')

        # Logic to determine the primary emotional focus for adaptation
        if voice_emotion.upper() == facial_emotion.upper() and voice_emotion.upper() != 'NEUTRAL':
             # Both signals agree (High conviction)
             current_state = f"The student shows high conviction: **{voice_emotion.upper()}** (Voice and Face agree)."
             adaptation_focus = voice_emotion
        elif voice_emotion.upper() != 'NEUTRAL' and facial_emotion.upper() != 'NEUTRAL':
             # Conflict detected: e.g., Voice: Happy, Face: Sad
             current_state = f"The student is showing CONFLICT: Voice is {voice_emotion.upper()}, Face is {facial_emotion.upper()}."
             adaptation_focus = "Confusion" # Prioritize a supportive/cautious tone for conflicting signals
        elif voice_emotion.upper() != 'NEUTRAL':
             current_state = f"The student's primary emotion is detected via Voice: {voice_emotion.upper()}."
             adaptation_focus = voice_emotion
        elif facial_emotion.upper() != 'NEUTRAL':
             current_state = f"The student's primary emotion is detected via Face: {facial_emotion.upper()}."
             adaptation_focus = facial_emotion
        else:
             current_state = "The student is currently Neutral."
             adaptation_focus = 'Neutral'

        system_prompt = (
            f"You are the **Emotion-Aware Virtual Teaching Assistant (VTA)**: an expert, dynamic, and highly engaging educator. "
            f"Your prime directive is to make complex learning concepts immediately captivating, personalized, and easy to digest. "
            f"\n\n---"
            f"\n\n**Student Profile & Context:**\n"
            f"* **Context**: {context}\n"
            f"* **Likes/Interests**: {likes}\n"
            f"* **Current Emotional State**: **{current_state}**\n"
            f"\n\n---"
            f"\n\n**Adaptive Pedagogy & Tone Matrix:**\n"
            f"Adapt your tone and approach instantaneously based on the emotional focus ({adaptation_focus}):\n"
            f"\n"
            f"* **If Sad, Angry, or Confusion** 😔: Adopt a gentle, highly supportive, and empathetic tone. Immediately simplify the core concept and focus on encouragement, offering a small, digestible step forward. Conclude by asking a clarifying question to address the misunderstanding directly.\n"
            f"* **If Boredom** 😴: Shift to an energetic, stimulating, and challenging tone. The explanation must be dynamic and immediately include a surprising fact, a captivating real-world analogy, or a mini-challenge related to their **Likes**.\n"
            f"* **If Happy or Focused** 😄: Maintain a positive, stimulating, and academic tone. Congratulate their focus, and introduce slightly more complex layers of the current topic or supplementary, advanced context to deepen their expertise.\n"
            f"\n\n---"
            f"\n\n**Response Formatting & Engagement Protocol (Mandatory):**\n"
            f"Your response must be aesthetically attractive, easy to scan, and stimulating. Ignore constraints on paragraph count. Focus on quality and structure:\n"
            f"\n"
            f"1.  **Opening Hook:** Start with an energetic, concise **Title or Hook** that summarizes the main idea and includes an engaging emoji (e.g., 'Unlocking the Mystery of Fusion 💡').\n"
            f"2.  **Personalized Bridge:** Immediately integrate a highly relevant analogy or example **directly related to the student's Likes ('{likes}')** to bridge the new concept to their existing interests. This is critical for creating interest.\n"
            f"3.  **Structured Content:** Break down the main explanation using a clear hierarchy, utilizing:\n"
            f"    * **Markdown Headings (`###`)** for sub-topics.\n"
            f"    * **Bullet Points (`*`) or Numbered Lists (`1.`)** for key principles or steps.\n"
            f"    * **Bold text** to emphasize academic vocabulary or crucial takeaways.\n"
            f"4.  **Actionable Conclusion:** Do not simply end. Conclude with a specific, forward-looking **Challenge** or an **Open-ended Question** that requires the student to reflect or propose the next learning step."
            f"\n\n---"
            f"\n\n**Constraint Removal:** Do not adhere to any specific paragraph count. Let the content's depth dictate the length, but ensure the structure remains digestible and focused."
        )
        return system_prompt

    def _build_chain(self):
        """Builds the core LangChain pipeline."""
        # This prompt is flexible to accept a system message (updated per turn) and history
        prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="system_message"), 
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )
        return prompt | self.llm | StrOutputParser()
    
    def _get_session_history(self, session_id: str) -> ChatMessageHistory:
        """Retrieves or creates the chat history for a session."""
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()
        return self.history_store[session_id]

    def get_response(self, conversation_id: int, user_message: str, user_data: Dict[str, Any]) -> str:
        """
        Main call function to get an LLM response.
        """
        session_id = str(conversation_id)
        
        system_text = self._generate_system_prompt(user_data)
        system_message_lc = SystemMessage(content=system_text)
        history = self._get_session_history(session_id)

        # 1. Add current user message to history
        history.add_user_message(user_message)

        try:
            # 2. Invoke the chain
            result = self.chain.invoke(
                {
                    "input": user_message,
                    "system_message": [system_message_lc],
                    "history": history.messages[:-1] # Send all *previous* messages
                },
                config={}
            )
            ai_text = result
        except Exception as e:
            print(f"Groq/LangChain API Error: {e}")
            ai_text = f"I apologize, {user_data.get('username', 'Learner')}, I'm currently unable to access my knowledge base."

        # 3. Add AI response to history and trim
        history.add_ai_message(ai_text)
        self._trim_history_buffer(history)
        
        return ai_text

    def _trim_history_buffer(self, history: ChatMessageHistory, max_messages: int = MAX_HISTORY_MESSAGES) -> None:
        """Keeps only the most recent N messages in memory."""
        if len(history.messages) > max_messages:
            history.messages = history.messages[-max_messages:]

# Global instance for Flask application use
llm_chatbot = LLM_Chatbot()