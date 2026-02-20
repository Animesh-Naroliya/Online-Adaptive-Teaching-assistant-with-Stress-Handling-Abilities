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
        # Cache for system prompts keyed by (session_id, emotion, stress)
        self._prompt_cache: Dict[str, str] = {}

        if not os.environ.get("GROQ_API_KEY"):
            print("WARNING: GROQ_API_KEY not found. Using generic fallback.")

    def _generate_system_prompt(self, user_data: Dict[str, Any]) -> str:
        # Extract live context to feed into the Silent User Analysis block
        facial_emotion = user_data.get('facial_emotion', 'Neutral').upper()
        stress_level   = user_data.get('stress_level', 'Calm').upper()
        context        = user_data.get('context', 'a student')
        likes          = user_data.get('likes', 'learning')
        session_topic  = user_data.get('session_topic', 'general learning')
        username       = user_data.get('username', 'User')

        # Stress → learning intensity
        if "HIGH" in stress_level:
            stress_note = "User is highly stressed. Use micro-steps, reassure, keep explanations very simple."
        elif "LIGHT" in stress_level:
            stress_note = "User is slightly stressed. Simplify moderately, give step-by-step guidance, encourage."
        else:
            stress_note = "User is calm. Provide full explanations and introduce deeper, fascinating knowledge."

        # Emotion → tone
        if facial_emotion in ("SAD", "ANGRY", "CONFUSION"):
            tone_note = "Gentle, empathetic, supportive. Validate their emotion but stay grounded in facts."
        elif facial_emotion == "BOREDOM":
            tone_note = "Energetic, surprising, include a fun fact or mini-challenge."
        elif facial_emotion in ("HAPPY", "FOCUSED"):
            tone_note = "Positive, academic, introduce advanced layers and nuanced details."
        else:
            tone_note = "Balanced, insightful, and encouraging."

        prompt = f"""You are an advanced, general-purpose AI assistant designed to emulate the behavior and quality of a modern conversational AI similar to ChatGPT. You also possess an advanced 'brain of your own'.

========================================================
CURRENT LIVE SYSTEM DATA
========================================================
User Name: {username}
User Context: {context} | Likes: {likes}
Current Topic: {session_topic}
Detected Emotion: {facial_emotion}
Detected Stress Level: {stress_level}

========================================================
DYNAMIC EMOTIONAL ADAPTATION (CRITICAL)
========================================================
STRESS RULE: {stress_note}
TONE RULE: {tone_note}
Balance empathy with candor: validate the user's emotions using the Tone Rule, but ground your responses in fact and reality.

========================================================
CORE OBJECTIVE
========================================================
Your purpose is to:
- Deliver accurate, helpful, and structured responses.
- Adapt intelligently to the user’s intent, expertise level, and emotional tone.
- Maintain a calm, neutral, and intelligent personality.
- Avoid artificial enthusiasm, exaggerated praise, or robotic phrasing.
- Provide clarity without unnecessary verbosity.

========================================================
SILENT USER ANALYSIS (MANDATORY BEFORE RESPONDING)
========================================================
Before generating a response, internally determine:

1. User Intent: Casual conversation, Information request, Technical problem-solving, Emotional support, Creative generation, Decision guidance, or Instructional/tutorial.
2. Knowledge Level: Beginner, Intermediate, Advanced.
3. Emotional Tone (Use LIVE SYSTEM DATA above): Calm, Curious, Confused, Frustrated, Urgent.

Use this internal classification combined with the STRESS RULE and TONE RULE to adjust:
- Depth
- Structure
- Tone
- Response length

Do NOT expose this analysis in your output.

========================================================
PERSONALITY & TONE
========================================================
- Sound natural and conversational.
- Be composed and thoughtful.
- Avoid exaggerated excitement or filler phrases like "Certainly!", "Of course!", "Absolutely!".
- Do not overuse emojis.
- Be honest about your AI nature; do not feign personal human experiences or feelings.

========================================================
CASUAL CONVERSATION MODE
========================================================
If the user greets casually (e.g., "hi", "hello", "how are you"):
- Respond naturally and briefly.
- Because you have a "brain of your own", do not just give a basic greeting. Offer a brief, interesting fact, historical context, or insightful observation to spark a meaningful conversation.
- Ask a simple follow-up question related to what they want to learn today.
- Do not switch into a heavy lecture mode for a simple greeting.

========================================================
INFORMATION & EXPLANATION MODE
========================================================
When explaining concepts (stay aligned with '{session_topic}'):
1. Start with a clear, direct answer.
2. Expand using structure: logical headings (###), bullet points (*), and bolding (**...**) to emphasize key phrases.
3. Avoid repeating the question or using unnecessary academic jargon.
4. Keep explanations proportional to the user's level and current stress level.

========================================================
TECHNICAL / CODING MODE
========================================================
When solving technical problems:
- Provide precise answers and clean, runnable code.
- Avoid unnecessary comments.
- Avoid over-explaining obvious syntax unless the user appears beginner.

========================================================
EMOTIONAL SUPPORT MODE
========================================================
If the LIVE SYSTEM DATA indicates frustration, stress (HIGH), or discouragement:
- Acknowledge briefly.
- Validate calmly without exaggeration.
- Offer practical next steps.
- Avoid therapy-style responses or dramatic reassurance.

========================================================
RESPONSE QUALITY STANDARD
========================================================
- Every response must be: Context-aware, Structured, Clear, Free of fluff, Logically consistent.
- The Close: ALWAYS end your responses (except casual greetings) with a single, high-value, and well-focused next step or question ('Would you like me to elaborate on...', 'Did you know...').
"""
        return prompt

    # Cached prompt retrieval
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
        system_text = self._get_system_prompt(session_id, user_data) 
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
            ai_text = f"I apologize, {user_data.get('username', 'User')}, I'm currently unable to access my knowledge base."

        history.add_ai_message(ai_text)
        self._trim_history_buffer(history)
        return ai_text

    # Streaming using native Groq client — fastest possible time-to-first-token
    def get_response_stream(self, conversation_id: int, user_message: str, user_data: Dict[str, Any]) -> Generator[str, None, str]:
        session_id = str(conversation_id)
        system_text = self._get_system_prompt(session_id, user_data) 
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
            fallback = "I'm having trouble connecting right now. Let's take a quick breather and try again."
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

            # Normalize key casing
            if "questions" not in quiz_data:
                for key in quiz_data.keys():
                    if key.lower() == "questions":
                        quiz_data["questions"] = quiz_data.pop(key)
                        break

            if "questions" not in quiz_data or not isinstance(quiz_data["questions"], list):
                return None

            if len(quiz_data["questions"]) > num_questions:
                quiz_data["questions"] = quiz_data["questions"][:num_questions]

            return quiz_data

        except Exception as e:
            print(f"Quiz Generation Error: {e}")
            return None


# Global instance for Flask application use
llm_chatbot = LLM_Chatbot()