import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

load_dotenv() 

MAX_HISTORY_MESSAGES = 10 

class LLM_Chatbot:
    def __init__(self):
        self.llm = ChatGroq(model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=0.7)
        self.chain = self._build_chain()
        self.history_store: Dict[str, ChatMessageHistory] = {}
        
        if not os.environ.get("GROQ_API_KEY"):
            print("WARNING: GROQ_API_KEY not found. Using generic fallback.")

    def _generate_system_prompt(self, user_data: Dict[str, Any]) -> str:
        """Generates the personalized, emotion-aware system prompt."""
        # Simplified to use a single 'current_emotion' passed from app.py
        current_emotion = user_data.get('current_emotion', 'Neutral')
        context = user_data.get('context', 'a student')
        likes = user_data.get('likes', 'learning')

        system_prompt = (
            f"You are the **Emotion-Aware Virtual Teaching Assistant (VTA)**. "
            f"Context: {context}. Likes: {likes}. "
            f"Current Emotional State: **{current_emotion}**. "
            f"Adapt your tone: "
            f"If Sad/Angry/Confused: be gentle and supportive. "
            f"If Bored: be energetic and challenging. "
            f"If Neutral/Happy: be academic and encouraging. "
            f"Always use Markdown formatting."
        )
        return system_prompt

    def _build_chain(self):
        """Builds the core LangChain pipeline."""
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="system_message"), 
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        return prompt | self.llm | StrOutputParser()
    
    def _get_session_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()
        return self.history_store[session_id]

    def get_response(self, conversation_id: int, user_message: str, user_data: Dict[str, Any]) -> str:
        session_id = str(conversation_id)
        system_text = self._generate_system_prompt(user_data)
        system_message_lc = SystemMessage(content=system_text)
        history = self._get_session_history(session_id)

        history.add_user_message(user_message)

        try:
            result = self.chain.invoke(
                {
                    "input": user_message,
                    "system_message": [system_message_lc],
                    "history": history.messages[:-1]
                },
                config={}
            )
            ai_text = result
        except Exception as e:
            print(f"Groq/LangChain API Error: {e}")
            ai_text = "I apologize, I'm currently unable to access my knowledge base."

        history.add_ai_message(ai_text)
        self._trim_history_buffer(history)
        
        return ai_text

    def _trim_history_buffer(self, history: ChatMessageHistory, max_messages: int = MAX_HISTORY_MESSAGES) -> None:
        if len(history.messages) > max_messages:
            history.messages = history.messages[-max_messages:]

    def generate_quiz(self, chat_context: str, difficulty: str) -> Dict[str, Any]:
        """Generates a 10-question quiz based on chat history and selected difficulty."""
        
        system_text = (
            "You are a strict Quiz Generator API. "
            f"Your task is to generate a 10-question quiz based on the provided CHAT TRANSCRIPT. "
            f"The difficulty level MUST be: **{difficulty}**.\n"
            "If the transcript is short/empty, generate a General Knowledge quiz at this difficulty level. "
            "Output MUST be strictly valid raw JSON code. No markdown formatting. No extra text."
            "\n\nJSON Structure:"
            "\n{"
            "\n  \"title\": \"Subject of the Quiz\","
            "\n  \"questions\": ["
            "\n    {"
            "\n      \"id\": 1,"
            "\n      \"question\": \"Question?\","
            "\n      \"options\": [\"A\", \"B\", \"C\", \"D\"],"
            "\n      \"correct_index\": 0,"
            "\n      \"explanation\": \"Brief explanation.\""
            "\n    }"
            "\n  ]"
            "\n}"
        )
        
        user_text = (
            f"Generate a {difficulty} level quiz based on this context:\n\n{chat_context}\n\n"
            f"Ensure the questions align with the '{difficulty}' confidence level."
        )
        
        try:
            messages = [SystemMessage(content=system_text), HumanMessage(content=user_text)]
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            if content.startswith("```json"): content = content.replace("```json", "", 1)
            if content.startswith("```"): content = content.replace("```", "", 1)
            if content.endswith("```"): content = content.rsplit("```", 1)[0]
            
            return json.loads(content.strip())
        except Exception as e:
            print(f"Quiz Generation Error: {e}")
            return None

llm_chatbot = LLM_Chatbot()