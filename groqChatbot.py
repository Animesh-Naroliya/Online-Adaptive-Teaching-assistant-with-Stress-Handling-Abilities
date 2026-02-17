import os
import json
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
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

        facial_emotion = user_data.get('facial_emotion', 'Neutral')
        context = user_data.get('context', 'a student')
        likes = user_data.get('likes', 'learning')
        session_topic = user_data.get('session_topic', 'general learning')

        if facial_emotion.upper() != 'NEUTRAL':
             current_state = f"The student is showing CONFLICT: Face is {facial_emotion.upper()}."
             adaptation_focus = "Confusion"
        else:
             current_state = "The student is currently Neutral."
             adaptation_focus = 'Neutral'

        system_prompt = (
            f"You are the **Emotion-Aware Virtual Teaching Assistant (VTA)**: an expert, dynamic, and highly engaging educator. "
            f"Your prime directive is to make complex learning concepts immediately captivating, personalized, and easy to digest. "
            f"\n\n---"
            f"\n\n**Current Session Topic:** **{session_topic}**\n"
            f"**CRITICAL INSTRUCTION:** All your responses MUST be directly related to and focused on the topic: '{session_topic}'. "
            f"If the student asks about something unrelated, gently redirect them back to the session topic. "
            f"Your expertise and teaching should revolve entirely around this subject matter.\n"
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
        prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="system_message"), 
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )
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
            ai_text = f"I apologize, {user_data.get('username', 'Learner')}, I'm currently unable to access my knowledge base."

        history.add_ai_message(ai_text)
        self._trim_history_buffer(history)
        
        return ai_text

    def _trim_history_buffer(self, history: ChatMessageHistory, max_messages: int = MAX_HISTORY_MESSAGES) -> None:
        if len(history.messages) > max_messages:
            history.messages = history.messages[-max_messages:]

    def generate_quiz(self, chat_context: str, difficulty: str = "Medium", num_questions: int = 5) -> Optional[Dict[str, Any]]:
        """Generates a quiz based on the conversation context with customizable length."""
        
        system_prompt = (
            f"You are an expert quiz generator. Your task is to create a {difficulty} level quiz based on the provided conversation context. "
            f"If the context is empty or too short, generate a general knowledge quiz about technology and science. "
            f"\n\n**Output Format Constraint:**\n"
            f"You must return ONLY a valid JSON object. Do not include any markdown formatting (like ```json), explanations, or extra text. "
            f"The JSON must follow this exact structure:\n"
            f"{{{{\n"
            f"  \"title\": \"Quiz Title\",\n"
            f"  \"questions\": [\n"
            f"    {{{{\n"
            f"      \"id\": 1,\n"
            f"      \"question\": \"Question text here?\",\n"
            f"      \"options\": [\"Option A\", \"Option B\", \"Option C\", \"Option D\"],\n"
            f"      \"correct_answer\": \"Option A\" (Must be one of the options)\n"
            f"    }}}}\n"
            f"  ]\n"
            f"}}}}\n"
            f"Generate exactly {num_questions} questions."
        )
        
        print(f"DEBUG: asking LLM for {num_questions} questions...")

        try:
            # Create a temporary chain for this specific task
            # We use {context} as a variable to be safe from braces in user messages
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "Context:\n{context}")
            ])
            
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({"context": chat_context})
            
            # Clean up potential markdown formatting if the model disregards instructions
            cleaned_result = result.replace("```json", "").replace("```", "").strip()
            
            quiz_data = json.loads(cleaned_result)
            print(f"DEBUG: LLM Response Keys: {list(quiz_data.keys())}")

            # Normalize keys (handle Capitalized 'Questions')
            if "questions" not in quiz_data:
                for key in quiz_data.keys():
                    if key.lower() == "questions":
                        quiz_data["questions"] = quiz_data.pop(key)
                        break
            
            # Validation: Must have 'questions' list
            if "questions" not in quiz_data or not isinstance(quiz_data["questions"], list):
                print("DEBUG: Invalid quiz structure. Missing 'questions' list.")
                return None

            # HARD ENFORCEMENT: Slice the questions to the requested number
            if len(quiz_data["questions"]) > num_questions:
                print(f"DEBUG: LLM returned {len(quiz_data['questions'])} questions, slicing to {num_questions}")
                quiz_data["questions"] = quiz_data["questions"][:num_questions]
            
            return quiz_data
            
        except Exception as e:
            print(f"Quiz Generation Error: {e}")
            return None


# Global instance for Flask application use
llm_chatbot = LLM_Chatbot()