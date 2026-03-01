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
            f"Use ONLY the subjects explicitly discussed in this chat context. "
            f"Do NOT introduce unrelated topics.\n\n"
            f"Return ONLY valid JSON (no markdown, no extra text) in this structure:\n"
            f'{{ "title": "...", "questions": [{{'
            f'"id":1,"question":"...","options":["A","B","C","D"],'
            f'"correct_answer":"A","explanation":"one-line clarification",'
            f'"misconception_map":{{"0":"why option 0 feels right but is wrong","1":"...","2":"...","3":"..."}}'
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

            # Parse defensively: models often prepend/append text around JSON.
            quiz_data = None
            try:
                quiz_data = json.loads(cleaned_result)
            except json.JSONDecodeError:
                extracted_json = self._extract_json_object(cleaned_result)
                if extracted_json:
                    quiz_data = json.loads(extracted_json)
                else:
                    print("Quiz parser: could not extract valid JSON; using fallback quiz.")
                    return self._build_fallback_quiz(difficulty, num_questions, chat_context)

            # Normalize key casing
            if "questions" not in quiz_data:
                for key in quiz_data.keys():
                    if key.lower() == "questions":
                        quiz_data["questions"] = quiz_data.pop(key)
                        break

            # Handle alternate top-level wrappers from model output.
            if "questions" not in quiz_data:
                for wrapper_key in ("quiz", "data", "payload"):
                    wrapped = quiz_data.get(wrapper_key)
                    if isinstance(wrapped, dict) and isinstance(wrapped.get("questions"), list):
                        quiz_data = wrapped
                        break

            # Handle top-level list output from model.
            if isinstance(quiz_data, list):
                quiz_data = {"title": f"{difficulty} Quiz", "questions": quiz_data}

            if "questions" not in quiz_data or not isinstance(quiz_data["questions"], list):
                print("Quiz parser: missing questions list; using fallback quiz.")
                return self._build_fallback_quiz(difficulty, num_questions, chat_context)

            normalized_questions: List[Dict[str, Any]] = []
            for index, q in enumerate(quiz_data["questions"], start=1):
                if not isinstance(q, dict):
                    continue

                options = q.get("options")
                if isinstance(options, dict):
                    # Accept {"A":"..","B":".."} format and keep consistent ordering.
                    ordered_keys = sorted(options.keys(), key=lambda x: str(x))
                    options = [str(options[k]) for k in ordered_keys]
                if not isinstance(options, list) or len(options) < 2:
                    continue

                question_text = q.get("question")
                if not question_text:
                    question_text = q.get("prompt")
                if not isinstance(question_text, str) or not question_text.strip():
                    continue

                # Normalize answer into a usable index for frontend logic.
                correct_index = q.get("correct_index")
                if not isinstance(correct_index, int):
                    alt_index = q.get("answer_index")
                    if isinstance(alt_index, int):
                        correct_index = alt_index
                if not isinstance(correct_index, int):
                    correct_answer = (
                        q.get("correct_answer")
                        or q.get("answer")
                        or q.get("correctOption")
                    )
                    if isinstance(correct_answer, str):
                        answer_clean = correct_answer.strip()
                        if len(answer_clean) == 1 and answer_clean.upper() in ("A", "B", "C", "D"):
                            correct_index = ord(answer_clean.upper()) - ord("A")
                        elif answer_clean.isdigit():
                            # Accept "1", "2", ... as 1-based answer indices.
                            numeric_index = int(answer_clean) - 1
                            if 0 <= numeric_index < len(options):
                                correct_index = numeric_index
                        else:
                            # Accept formats like "A) ...", "Option B", etc.
                            for letter in ("A", "B", "C", "D"):
                                if letter in answer_clean.upper() and len(options) >= (ord(letter) - ord("A") + 1):
                                    correct_index = ord(letter) - ord("A")
                                    break
                            try:
                                if not isinstance(correct_index, int):
                                    correct_index = options.index(answer_clean)
                            except ValueError:
                                lower_options = [str(opt).strip().lower() for opt in options]
                                if answer_clean.lower() in lower_options:
                                    correct_index = lower_options.index(answer_clean.lower())
                    elif isinstance(correct_answer, int):
                        correct_index = correct_answer

                if not isinstance(correct_index, int) or not (0 <= correct_index < len(options)):
                    continue

                explanation = q.get("explanation")
                if not isinstance(explanation, str) or not explanation.strip():
                    explanation = f'"{options[correct_index]}" is correct because it best matches the concept discussed in the chat.'

                misconception_raw = q.get("misconception_map")
                misconception_map: Dict[str, str] = {}
                if isinstance(misconception_raw, dict):
                    for k, v in misconception_raw.items():
                        if isinstance(v, str) and v.strip():
                            key = str(k).strip()
                            misconception_map[key] = v.strip()
                elif isinstance(q.get("misconceptions"), list):
                    for opt_i, text in enumerate(q.get("misconceptions")):
                        if isinstance(text, str) and text.strip():
                            misconception_map[str(opt_i)] = text.strip()

                if not misconception_map:
                    for opt_i, opt_text in enumerate(options):
                        if opt_i == correct_index:
                            continue
                        misconception_map[str(opt_i)] = f'"{opt_text}" sounds close but misses the key relationship in this topic.'

                normalized_questions.append({
                    "id": q.get("id", index),
                    "question": question_text.strip(),
                    "options": options,
                    "correct_index": correct_index,
                    "correct_answer": options[correct_index],
                    "explanation": explanation.strip(),
                    "misconception_map": misconception_map
                })

            if not normalized_questions:
                print("Quiz parser: no valid normalized questions; using fallback quiz.")
                return self._build_fallback_quiz(difficulty, num_questions, chat_context)

            quiz_data["questions"] = normalized_questions

            if len(quiz_data["questions"]) > num_questions:
                quiz_data["questions"] = quiz_data["questions"][:num_questions]

            if "title" not in quiz_data or not isinstance(quiz_data["title"], str):
                quiz_data["title"] = f"{difficulty} Quiz"

            return quiz_data

        except Exception as e:
            print(f"Quiz Generation Error: {e}")
            return self._build_fallback_quiz(difficulty, num_questions, chat_context)

    def _extract_json_object(self, text: str) -> Optional[str]:
        """Extract the first balanced JSON object from model output text."""
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return None

    def _build_fallback_quiz(self, difficulty: str, num_questions: int, chat_context: str) -> Dict[str, Any]:
        """Return a guaranteed-valid quiz payload when AI formatting is unusable."""
        topic_hint = "General Knowledge"
        if isinstance(chat_context, str) and chat_context.strip():
            for line in chat_context.splitlines():
                if line.lower().startswith("session topic:"):
                    extracted = line.split(":", 1)[1].strip()
                    if extracted:
                        topic_hint = extracted
                        break
            if topic_hint == "General Knowledge":
                words = [w for w in chat_context.replace("\n", " ").split(" ") if w.strip()]
                if words:
                    topic_hint = " ".join(words[:4]).strip(":,.!?") or topic_hint

        bank = [
            {
                "question": f"Which statement best matches the concept of {topic_hint}?",
                "options": [
                    f"A core idea from {topic_hint}",
                    "An unrelated random claim",
                    "A math-only identity",
                    "A UI color palette rule"
                ],
                "correct_index": 0,
                "explanation": f"The correct choice directly reflects the core idea discussed for {topic_hint}.",
                "misconception_map": {
                    "1": "It may feel specific, but it is unrelated to the topic.",
                    "2": "It sounds technical, but not from this chat context.",
                    "3": "It confuses interface design with subject understanding."
                }
            },
            {
                "question": f"In the discussion about {topic_hint}, which approach is most context-relevant?",
                "options": [
                    "Ignoring the topic entirely",
                    f"Using examples directly tied to {topic_hint}",
                    "Switching to an unrelated subject",
                    "Avoiding all definitions"
                ],
                "correct_index": 1,
                "explanation": "Topic-linked examples test understanding more reliably than unrelated shifts.",
                "misconception_map": {
                    "0": "Skipping context can feel simpler but breaks comprehension.",
                    "2": "Changing subject feels productive but avoids the learning goal.",
                    "3": "Avoiding definitions removes conceptual clarity."
                }
            },
            {
                "question": f"What should be prioritized when revising {topic_hint} from this chat?",
                "options": [
                    "Memorizing random facts",
                    "Reading only UI code",
                    "Reviewing key terms and explanations from the conversation",
                    "Skipping topic-specific details"
                ],
                "correct_index": 2,
                "explanation": "Reviewing the same concepts and explanations strengthens topic retention.",
                "misconception_map": {
                    "0": "Random facts feel broad but are not aligned with your discussion.",
                    "1": "UI-only review may look useful but ignores subject concepts.",
                    "3": "Skipping details creates shallow understanding."
                }
            },
            {
                "question": f"Which quiz question is most aligned with {topic_hint}?",
                "options": [
                    f"A question directly about {topic_hint}",
                    "A question about unrelated sports scores",
                    "A question about random geography",
                    "A question about movie plots"
                ],
                "correct_index": 0,
                "explanation": "Alignment means the question must test what was actually discussed.",
                "misconception_map": {
                    "1": "Current events can be engaging but are off-topic here.",
                    "2": "General knowledge is useful but not this session’s focus.",
                    "3": "Narrative recall does not assess the target concept."
                }
            },
            {
                "question": f"To check understanding of {topic_hint}, what is the best method?",
                "options": [
                    "Avoid topic vocabulary",
                    "Ask context-based conceptual questions",
                    "Use only yes/no questions with no context",
                    "Never validate answers"
                ],
                "correct_index": 1,
                "explanation": "Conceptual, context-based questions reveal whether ideas are truly understood.",
                "misconception_map": {
                    "0": "Removing key terms can simplify wording but hides understanding gaps.",
                    "2": "Binary questions are quick but weak for depth checks.",
                    "3": "Without validation, errors remain uncorrected."
                }
            },
            {
                "question": f"When answering questions on {topic_hint}, which is preferred?",
                "options": [
                    "Context-aware explanations",
                    "Completely off-topic answers",
                    "Only unrelated trivia",
                    "No reasoning at all"
                ],
                "correct_index": 0,
                "explanation": "Reasoning tied to context demonstrates genuine comprehension.",
                "misconception_map": {
                    "1": "Off-topic responses may sound fluent but fail the objective.",
                    "2": "Trivia knowledge does not replace concept mastery.",
                    "3": "No reasoning prevents diagnosis of misunderstanding."
                }
            },
            {
                "question": f"Which option indicates strong understanding of {topic_hint}?",
                "options": [
                    "Cannot relate any concept",
                    "Can connect concepts to examples discussed",
                    "Avoids all key terms",
                    "Answers every question randomly"
                ],
                "correct_index": 1,
                "explanation": "Linking ideas to examples from the chat is a strong signal of understanding.",
                "misconception_map": {
                    "0": "Admitting no relation is honest but shows lack of conceptual grasp.",
                    "2": "Avoiding terms can hide uncertainty rather than resolve it.",
                    "3": "Random answers can score by luck but show no mastery."
                }
            },
            {
                "question": f"What is a good next step after learning {topic_hint}?",
                "options": [
                    "Switch topics immediately without review",
                    "Practice with topic-focused quiz questions",
                    "Ignore mistakes",
                    "Avoid asking follow-up questions"
                ],
                "correct_index": 1,
                "explanation": "Targeted practice plus feedback converts short-term recall into stable understanding.",
                "misconception_map": {
                    "0": "Moving on quickly feels efficient but weakens retention.",
                    "2": "Ignoring mistakes repeats the same misunderstanding.",
                    "3": "No follow-up prevents clarification of weak points."
                }
            },
        ]

        if num_questions < 1:
            num_questions = 1
        selected = bank[:num_questions] if num_questions <= len(bank) else (bank * ((num_questions // len(bank)) + 1))[:num_questions]

        questions = []
        for idx, q in enumerate(selected, start=1):
            questions.append({
                "id": idx,
                "question": q["question"],
                "options": q["options"],
                "correct_index": q["correct_index"],
                "correct_answer": q["options"][q["correct_index"]],
                "explanation": q.get("explanation", "This option best matches the discussed concept."),
                "misconception_map": q.get("misconception_map", {})
            })

        return {
            "title": f"{difficulty} Quiz - {topic_hint}",
            "questions": questions,
        }


# Global instance for Flask application use
llm_chatbot = LLM_Chatbot()
