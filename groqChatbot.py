import os
import json
from typing import Dict, Any, List, Optional, Generator
from dotenv import load_dotenv
from groq import Groq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

load_dotenv()

MAX_HISTORY_MESSAGES = 10


class LLM_Chatbot:
    def __init__(self):
        self.llm = ChatGroq(
            model=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0.7,
            streaming=True   # Enable streaming at model level
        )
        self.chain = self._build_chain()
        self.history_store: Dict[str, ChatMessageHistory] = {}
        # FIX 3: Cache for system prompts keyed by (session_id, emotion, stress)
        self._prompt_cache: Dict[str, str] = {}

        if not os.environ.get("GROQ_API_KEY"):
            print("WARNING: GROQ_API_KEY not found. Using generic fallback.")

    # FIX 1: Shortened, concise system prompt (~60% smaller, same adaptive logic)
    def _generate_system_prompt(self, user_data: Dict[str, Any]) -> str:
        facial_emotion = user_data.get('facial_emotion', 'Neutral').upper()
        stress_level   = user_data.get('stress_level', 'Calm').upper()
        context        = user_data.get('context', 'a student')
        likes          = user_data.get('likes', 'learning')
        session_topic  = user_data.get('session_topic', 'general learning')

        # Stress → learning intensity
        if "HIGH" in stress_level:
            stress_note = "Use micro-steps, reassure, keep it very simple."
        elif "LIGHT" in stress_level:
            stress_note = "Simplify moderately, give step-by-step guidance, encourage."
        else:
            stress_note = "Full explanations, introduce deeper knowledge."

        # Emotion → tone
        if facial_emotion in ("SAD", "ANGRY", "CONFUSION"):
            tone_note = "Gentle, empathetic, supportive. End with a clarifying question."
        elif facial_emotion == "BOREDOM":
            tone_note = "Energetic, surprising, include a fun fact or mini-challenge."
        elif facial_emotion in ("HAPPY", "FOCUSED"):
            tone_note = "Positive, academic, introduce advanced layers."
        else:
            tone_note = "Balanced and encouraging."

        prompt = (
            f"You are the Emotion-Aware Virtual Teaching Assistant (VTA). "
            f"TOPIC: '{session_topic}' — stay strictly on this topic; redirect off-topic questions.\n"
            f"STUDENT: {context} | Likes: {likes} | Emotion: {facial_emotion} | Stress: {stress_level}\n\n"
            f"STRESS RULE: {stress_note}\n"
            f"TONE RULE: {tone_note}\n\n"
            f"RESPONSE FORMAT:\n"
            f"1. Hook: 1-line engaging title + emoji.\n"
            f"2. Bridge: 1 analogy tied to student's likes ('{likes}').\n"
            f"3. Content: Use ### headings, bullet points, bold key terms.\n"
            f"4. Close: End with a challenge or open question.\n"
            f"Be thorough but scannable. No filler sentences."
        )
        return prompt

    # FIX 3: Cached prompt retrieval
    def _get_system_prompt(self, session_id: str, user_data: Dict[str, Any]) -> str:
        emotion = user_data.get('facial_emotion', 'Neutral').upper()
        stress  = user_data.get('stress_level', 'Calm').upper()
        cache_key = f"{session_id}|{emotion}|{stress}"

        if cache_key not in self._prompt_cache:
            self._prompt_cache[cache_key] = self._generate_system_prompt(user_data)
            # Keep cache bounded to avoid memory growth
            if len(self._prompt_cache) > 200:
                oldest = next(iter(self._prompt_cache))
                del self._prompt_cache[oldest]

        return self._prompt_cache[cache_key]

    def _build_chain(self):
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

    def _trim_history_buffer(self, history: ChatMessageHistory) -> None:
        if len(history.messages) > MAX_HISTORY_MESSAGES:
            history.messages = history.messages[-MAX_HISTORY_MESSAGES:]

    # Original blocking method (kept for compatibility / quiz)
    def get_response(self, conversation_id: int, user_message: str, user_data: Dict[str, Any]) -> str:
        session_id = str(conversation_id)
        system_text = self._get_system_prompt(session_id, user_data)  # FIX 3
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
            ai_text = f"I apologize, {user_data.get('username', 'Learner')}, I'm currently unable to access my knowledge base."

        history.add_ai_message(ai_text)
        self._trim_history_buffer(history)
        return ai_text

    # FIX 2: Streaming using native Groq client — fastest possible time-to-first-token
    def get_response_stream(self, conversation_id: int, user_message: str, user_data: Dict[str, Any]) -> Generator[str, None, str]:
        session_id = str(conversation_id)
        system_text = self._get_system_prompt(session_id, user_data)  # FIX 3
        history = self._get_session_history(session_id)
        history.add_user_message(user_message)

        # Build messages list for native Groq API (avoids LangChain overhead)
        messages = [{"role": "system", "content": system_text}]
        for msg in history.messages[:-1]:  # exclude the just-added user message
            role = "user" if msg.type == "human" else "assistant"
            messages.append({"role": role, "content": msg.content})
        messages.append({"role": "user", "content": user_message})

        full_response = ""
        try:
            # Use the native Groq client directly for minimum latency
            groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            stream = groq_client.chat.completions.create(
                model=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
                messages=messages,
                temperature=0.7,
                stream=True
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_response += delta
                    yield delta
        except Exception as e:
            print(f"Groq Native Streaming Error: {e}")
            fallback = "I'm having trouble connecting right now. Please try again."
            full_response = fallback
            yield fallback

        history.add_ai_message(full_response)
        self._trim_history_buffer(history)

    def generate_quiz(self, chat_context: str, difficulty: str = "Medium", num_questions: int = 5) -> Optional[Dict[str, Any]]:
        """Generates a quiz based on the conversation context."""
        system_prompt = (
            f"You are a quiz generator. Create a {difficulty} quiz from the context below. "
            f"If context is too short, generate a general technology/science quiz.\n\n"
            f"Return ONLY valid JSON (no markdown, no extra text) in this structure:\n"
            f'{{ "title": "...", "questions": [{{'
            f'"id":1,"question":"...","options":["A","B","C","D"],"correct_answer":"A"'
            f'}}] }}\n'
            f"Generate exactly {num_questions} questions."
        )

        print(f"DEBUG: Generating quiz — {num_questions} questions, difficulty={difficulty}")

        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "Context:\n{context}")
            ])
            # Use non-streaming LLM for quiz (we need the full JSON at once)
            llm_no_stream = ChatGroq(
                model=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
                temperature=0.3,
                streaming=False
            )
            chain = prompt | llm_no_stream | StrOutputParser()
            result = chain.invoke({"context": chat_context})

            cleaned_result = result.replace("```json", "").replace("```", "").strip()
            quiz_data = json.loads(cleaned_result)
            print(f"DEBUG: LLM Response Keys: {list(quiz_data.keys())}")

            # Normalize key casing
            if "questions" not in quiz_data:
                for key in quiz_data.keys():
                    if key.lower() == "questions":
                        quiz_data["questions"] = quiz_data.pop(key)
                        break

            if "questions" not in quiz_data or not isinstance(quiz_data["questions"], list):
                print("DEBUG: Invalid quiz structure.")
                return None

            if len(quiz_data["questions"]) > num_questions:
                quiz_data["questions"] = quiz_data["questions"][:num_questions]

            return quiz_data

        except Exception as e:
            print(f"Quiz Generation Error: {e}")
            return None


# Global instance for Flask application use
llm_chatbot = LLM_Chatbot()