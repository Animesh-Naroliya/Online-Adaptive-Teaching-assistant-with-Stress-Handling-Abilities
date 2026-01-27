# 🤖 EVTA: Emotion-Aware Virtual Teaching Assistant

## ✨ Overview: The Future of Personalized Learning

The **Emotion-Aware Virtual Teaching Assistant (EVTA)** is a pioneering system designed to bridge the emotional gap in online education. We move beyond standard chatbots by integrating real-time **multimodal emotion recognition**—combining facial expressions and voice cues—to create a truly personalized and supportive digital tutor.

By understanding a student's cognitive and emotional state (e.g., **confusion, frustration, focus, confidence**), the EVTA dynamically adjusts its teaching style, timing interventions, and optimizing the learning path to maximize comprehension and minimize dropout rates.



---

## 🚀 Key Features & Multimodal Intelligence

Our assistant relies on a robust, two-pronged sensory approach for superior accuracy:

### 👁️ Visual & Auditory Sensing

* **Real-time Emotion Detection:** Utilizes deep learning models (**DeepFace for visual, Librosa for audio**) to process streams from the webcam and microphone, detecting granular emotional states (e.g., Happy, Sad, Confused, Frustrated).
* **Multimodal Fusion Logic:** Combines facial and voice inputs for a reliable, context-aware assessment. This ensures the assistant's response is accurate even when one signal is noisy (e.g., background noise in audio is overruled by focused facial cues).

### 💡 Adaptive Pedagogy

* **Dynamic Adaptive Feedback:** Provides tailored, real-time responses based on the fused emotional state.
    * **If Confused:** Re-explains the concept using a simpler analogy or provides stepwise hints.
    * **If Frustrated:** Suggests a short, engaging break or a change of topic/example.
    * **If Confident:** Introduces more complex problems or advanced supplementary material.
* **Interactive Learning Modules:** Includes built-in lessons on various subjects, designed with a focus on human-centered design principles for maximum engagement.

### 📈 Data & Analytics

* **Progress Tracking & Logging:** All interactions, including the detected emotional state, the system's intervention, and the student's subsequent emotional shift, are logged (e.g., in **SQLite**) to help educators understand engagement patterns and identify persistent struggle areas.

---

## 💻 Getting Started

These instructions will guide you through setting up and running the EVTA on your local machine for development and testing.

### Prerequisites

You will need the following installed on your system:

* **Python 3.8+**
* **Git**
* **A stable webcam and microphone**

### Installation Steps

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/emotion-aware-teaching-assistant.git](https://github.com/your-username/emotion-aware-teaching-assistant.git)
    cd emotion-aware-teaching-assistant
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv venv
    
    # On macOS/Linux:
    source venv/bin/activate
    
    # On Windows:
    venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: This includes libraries like OpenCV, DeepFace, Librosa, and the required LLM interface/API access.)*

---

## ▶️ Usage

To launch the EVTA application, run the main script from the project's root directory while your virtual environment is active:

```bash
python app.py
```

### **Project Team**

  * Animesh Naroliya
  * Kriti Upadhyay
  * Lithiga Jayaprakash
  * Reeya Ottalwar 
  
### **Thank You**
