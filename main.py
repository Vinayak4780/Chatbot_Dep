# import sys
# import multiprocessing as mp

# # ─── 1) FORCE 'spawn' FOR CUDA‐SAFE SUBPROCESSING ───────────────────────────
# if __name__ == "__main__":
#     mp.set_start_method("spawn", force=True)

# # ─── 2) CORE DEPENDENCIES ────────────────────────────────────────────────────
# import spacy
# from textblob import TextBlob
# from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import contractions
# import emoji
# from nltk.corpus import stopwords
# import string
# import re
# # ─── 3) IMPORT YOUR CREWAI AGENTS ────────────────────────────────────────────
# from agents.admission  import admission_agent,admission_task
# from agents.program    import program_agent, program_task
# from agents.faculties  import faculty_agent, faculty_task
# from agents.member     import board_agent, board_task
# from agents.student    import student_agent,    student_task
# from agents.default_agent import default_agent, default_task
# from agents.campus     import campus_agent, campus_task

# # ─── 4) PREPROCESSING ───────────────────────────────────────────────────────
# nlp = spacy.load("en_core_web_sm")
# STOP = set(stopwords.words("english"))

# def preprocess_query(text: str) -> str:
#     # 1) Expand common English contractions
#     text = contractions.fix(text)
#     # 2) Strip out emojis and non-printables
#     text = emoji.replace_emoji(text, replace="")
#     text = "".join(ch for ch in text if ch in string.printable)
#     # 3) Lowercase & remove HTML tags
#     text = re.sub(r"<[^>]+>", " ", text).lower()
#     # 4) Fix obvious spelling mistakes
#     text = str(TextBlob(text).correct())
#     # 5) Lemmatize & drop stopwords/punctuation/numbers
#     doc = nlp(text)
#     tokens = []
#     for tok in doc:
#         if tok.is_stop or tok.is_punct or tok.like_num:
#             continue
#         lemma = tok.lemma_.strip()
#         if len(lemma) > 1:
#             tokens.append(lemma)
#     return " ".join(tokens)


# # ─── 5) EMBEDDINGS & AGENT PROFILES ─────────────────────────────────────────
# embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# INTENT_AGENTS = {
#     "admission": admission_agent,
#     "program":   program_agent,
#     "faculties": faculty_agent,
#     "member":    board_agent,
#     "student":   student_agent,
#     "campus":    campus_agent,
# }
# INTENT_PROFILES = {
#     label: f"{agent.backstory} {agent.goal}"
#     for label, agent in INTENT_AGENTS.items()
# }

# # Precompute embeddings for each agent profile
# INTENT_EMBS = {
#     label: np.array(embedder.embed_documents([profile])[0])
#     for label, profile in INTENT_PROFILES.items()
# }

# SIMILARITY_THRESHOLD = 0.3  # tune this as needed

# # ─── 6) EMBEDDING‐BASED INTENT CLASSIFICATION ───────────────────────────────
# def classify_intent(query: str) -> str:
#     q_emb = np.array(embedder.embed_query(query))
#     sims = {
#         label: cosine_similarity(q_emb.reshape(1, -1), emb.reshape(1, -1))[0][0]
#         for label, emb in INTENT_EMBS.items()
#     }
#     best_label, best_score = max(sims.items(), key=lambda kv: kv[1])
#     return best_label if best_score >= SIMILARITY_THRESHOLD else "general_fallback"

# # ─── 7) MAIN HANDLER WITH FALLBACK TO DEFAULT_AGENT ─────────────────────────
# from typing import Optional
# def handle_query(user_query: str) -> str:
#     # 1) Preprocess & classify
#     clean_q = preprocess_query(user_query)
#     intent  = classify_intent(clean_q)

#     # 2) Choose the intent‐specific callback (or the default QA if nobody matches)
#     HANDLER_FUNCS = {
#         "admission": admission_task.callback,
#         "program":   program_task.callback,
#         "faculties": faculty_task.callback,
#         "member":    board_task.callback,
#         "student":   student_task.callback,
#         "campus":    campus_task.callback,
#     }
#     specialist = HANDLER_FUNCS.get(intent, None)

#     # 3) Wrap the input so it has a .input attribute
#     class Q:
#         def __init__(self, text): self.input = text

#     q = Q(user_query)

#     # 4) Run both handlers
#     #    - Specialist (if any)
#     #    - Default QA
#     # We’ll catch exceptions so one failure doesn’t kill the whole request.
#     try:
#         spec_result = specialist(q) if specialist else None
#     except Exception:
#         spec_result = None

#     try:
#         default_result = default_task.callback(q)
#     except Exception:
#         default_result = "Sorry, I'm having trouble right now."

#     # 5) Decide which to return
#     def is_valid_answer(text: Optional[str]) -> bool:
#         if not text:
#             return False
#         lower = text.lower().strip()
#         return not (
#             lower.startswith("i'm sorry")
#             or "don't have that information" in lower
#         )

#     if is_valid_answer(spec_result):
#         return spec_result
#     else:
#         return default_result


# # ─── 8) OPTIONAL CLI CONCURRENCY WRAPPER ──────────────────────────────────
# process_pool = ProcessPoolExecutor(max_workers=4)
# thread_pool  = ThreadPoolExecutor(max_workers=20)

# def query_in_process(q: str) -> str:
#     return process_pool.submit(handle_query, q).result()

# if __name__ == "__main__":
#     print("DSEU Unified Assistant (type 'exit' or 'quit' to stop')")
#     runner = query_in_process

#     for line in sys.stdin:
#         q = line.strip()
#         if q.lower() in ("exit", "quit"):
#             break
#         print(f"\n🤖 {runner(q)}\n")

#!/usr/bin/env python3

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import sys
import re
import string
import emoji
import contractions
from textblob import TextBlob
import spacy
import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
from autocorrect import Speller

# create an English speller
speller = Speller(lang='en')

from langchain_community.embeddings import HuggingFaceEmbeddings
from crewai import Task

# Import your CrewAI agents and tasks
from agents.admission import admission_task
from agents.program import program_task
from agents.faculties import faculty_task
from agents.member import board_task
from agents.student import student_task
from agents.campus import campus_task
from agents.default_agent import default_task

from typing import Optional

# ─── 1) PREPROCESSING ───────────────────────────────────────────────────────
nlp = spacy.load("en_core_web_sm")
STOP = set(stopwords.words("english"))

def preprocess_query(text: str) -> str:
    text = speller(text)  # Correct spelling
    text = contractions.fix(text)  # Expand contractions
    text = emoji.replace_emoji(text, replace="")  # Remove emojis
    text = "".join(ch for ch in text if ch in string.printable)
    text = re.sub(r"<[^>]+>", " ", text).lower()
    text = str(TextBlob(text).correct())
    doc = nlp(text)
    tokens = [tok.lemma_.strip() for tok in doc if not (tok.is_stop or tok.is_punct or tok.like_num) and len(tok.lemma_.strip()) > 1]
    return " ".join(tokens)

# ─── 2) EMBEDDINGS & INTENT PROFILES ────────────────────────────────────────
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

INTENT_AGENTS = {
    "admission": admission_task,
    "program":   program_task,
    "faculties": faculty_task,
    "member":    board_task,
    "student":   student_task,
    "campus":    campus_task,
}

# Combine backstory and goal from each agent for embedding
INTENT_PROFILES = {
    label: f"{agent.agent.backstory} {agent.agent.goal}"
    for label, agent in INTENT_AGENTS.items()
}

# Precompute profile embeddings
INTENT_EMBS = {
    label: np.array(embedder.embed_documents([profile])[0])
    for label, profile in INTENT_PROFILES.items()
}
SIMILARITY_THRESHOLD = 0.3

# ─── 3) INTENT CLASSIFICATION ──────────────────────────────────────────────
def classify_intent(query: str) -> str:
    q_emb = np.array(embedder.embed_query(query))
    sims = {
        label: cosine_similarity(q_emb.reshape(1, -1), emb.reshape(1, -1))[0][0]
        for label, emb in INTENT_EMBS.items()
    }
    best_label, best_score = max(sims.items(), key=lambda kv: kv[1])
    return best_label if best_score >= SIMILARITY_THRESHOLD else "general_fallback"

# ─── 4) MAIN HANDLER WITH FALLBACK ─────────────────────────────────────────
def handle_query(user_query: str) -> str:
    clean_q = preprocess_query(user_query)
    intent  = classify_intent(clean_q)

    # Select specialist callback
    HANDLER_FUNCS = {
        label: task.callback
        for label, task in INTENT_AGENTS.items()
    }
    specialist = HANDLER_FUNCS.get(intent)

    # Simple wrapper to match Task.callback signature
    class Q:
        def __init__(self, text): self.input = text
    q = Q(user_query)

    # Run specialist
    try:
        spec_result = specialist(q) if specialist else None
    except Exception:
        spec_result = None

    # Always run default QA
    try:
        default_result = default_task.callback(q)
    except Exception:
         spec_result = specialist(q) if specialist else None

    # Check validity
    def is_valid_answer(text: Optional[str]) -> bool:
        if not text:
            return False
        lower = text.lower().strip()
        return not (lower.startswith("i'm sorry") or "don't have that information" in lower)

    return spec_result if is_valid_answer(spec_result) else default_result

# ─── 5) CONCURRENCY WRAPPER ─────────────────────────────────────────────────
model_pool = ThreadPoolExecutor(max_workers=4)

def query_in_process(q: str) -> str:
    return model_pool.submit(handle_query, q).result()

if __name__ == "__main__":
    print("DSEU Unified Assistant (type 'exit' or 'quit' to stop)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            break
        print(f"\n🤖 {query_in_process(user_input)}\n")
