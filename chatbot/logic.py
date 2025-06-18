import random
import numpy as np
import nltk
from nltk import pos_tag
from nltk.chunk import RegexpParser
from sqlalchemy import text
from chatbot.model import model, intents, classes, words
from chatbot.preprocess import bow
from chatbot.db import engine

limit = random.randint(5, 7)
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def extract_noun_phrases(text):
    book_like_words = {
        'book', 'books', 'novel', 'novels', 'literature',
        'textbook', 'textbooks', 'volume', 'volumes',
        'publication', 'publications', 'work', 'works',
        'guide', 'guides', 'story', 'stories',
        'tale', 'tales', 'manual', 'manuals',
        'reference', 'reading'
    }

    tokens = nltk.word_tokenize(text)
    tagged = pos_tag(tokens)

    grammar = r"""
        NP: {<DT>?<JJ>*<NN.*>+(<IN><NN.*>+)*}
    """
    cp = RegexpParser(grammar)
    tree = cp.parse(tagged)

    noun_phrases = []
    for subtree in tree.subtrees():
        if subtree.label() == "NP":
            np = " ".join(word for word, tag in subtree)
            tokens_lower = np.lower().split()
            if len(tokens_lower) < 2:
                continue
            if tokens_lower[0] == 'i':
                continue
            if tokens_lower[-1] in book_like_words:
                np = " ".join(tokens_lower[:-1])
            noun_phrases.append(np.strip())
    return noun_phrases

def extract_book_title_from_db(user_input):
    noun_phrases = extract_noun_phrases(user_input)
    if not noun_phrases:
        return []

    results = []
    for phrase in noun_phrases:
        query = text(f"""
            SELECT title, description, authors
            FROM books
            WHERE LOWER(title) LIKE :phrase
            LIMIT {limit}
        """)
        with engine.connect() as conn:
            books = conn.execute(query, {"phrase": f"%{phrase.lower()}%"}).fetchall()
            for b in books:
                results.append({
                    "Book": b.title,
                    "Authors": b.authors,
                    "Feedback": b.description
                })
    return results

def get_books_by_category(tag):
    query = text("""
        SELECT b.title, b.description, b.authors
        FROM books b
        JOIN book_category bc ON b.id = bc.book_id
        JOIN categories c ON bc.category_id = c.id
        WHERE LOWER(c.name) = :tag
    """)
    with engine.connect() as conn:
        rows = conn.execute(query, {"tag": tag.lower()}).fetchall()
        rows = list(rows)

        if not rows:
            return []

        num_samples = min(len(rows), random.randint(5, 7))
        sampled = random.sample(rows, num_samples)

        return [
            {"Book": r.title, "Authors": r.authors, "Feedback": r.description}
            for r in sampled
        ]

def get_answer_for_general_questions(tag):
    query = text("""
        SELECT g.tag, g.answer
        FROM general_questions g
        WHERE LOWER(g.tag) = :tag
    """)
    with engine.connect() as conn:
        rows = conn.execute(query, {"tag": tag.lower()}).fetchall()
        rows = list(rows)

        if not rows:
            return []
        return [r.answer for r in rows]
    
def getResponse(ints, intents_json, user_input):
    if not ints:
        return [{"message": "I don't understand your question. Can you be more specific?"}]

    tag = ints[0]['intent']
    print("Detected tag:", tag)

    simple_tags = ['greeting', 'goodbye', 'thanks', 'book_search', 'document_search', 'Q&A', 'loan_period', 'penalty_policy', 'max_quantity_borrowed_books']

    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            if tag in simple_tags:
                return [{"message": random.choice(get_answer_for_general_questions(tag))}]

            elif tag == "specific_book_search":
                matched = extract_book_title_from_db(user_input)
                if not matched:
                    return [{"message": "Sorry, I couldn't find that book in our library."}]
                return matched

            else:
                books = get_books_by_category(tag)
                if not books:
                    return [{"message": f"Sorry, I couldn't find any books under '{tag}'."}]
                return books

    return [{"message": "Sorry, something went wrong."}]

def chatbot_response(msg):
    ints = predict_class(msg, model)
    return getResponse(ints, intents, msg)
