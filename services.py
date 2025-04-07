
import json
import re
from dotenv import load_dotenv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import BertForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
import nltk
from textblob_ar import TextBlob
from farasa.ner import FarasaNamedEntityRecognizer
from farasa.pos import FarasaPOSTagger
from farasa.stemmer import FarasaStemmer
from farasa.segmenter import FarasaSegmenter
import pyarabic.araby as araby
from schemas import TextAnalysis
from google import genai
import os
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

load_dotenv()


checkpoint_path = "./afnd-model"
model_name = "aubmindlab/bert-base-arabertv02"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = BertForSequenceClassification.from_pretrained(checkpoint_path)
model.eval()

# Gemini Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
google_search_tool = Tool(
    google_search=GoogleSearch()
)
model_id = "gemini-2.0-flash"

# Farasa
segmenter = FarasaSegmenter()
stemmer = FarasaStemmer()


def analyse_text(text: str) -> TextAnalysis:
    most_frequent_words = get_most_frequent_words(text)
    word_count, sent_count = get_word_and_sentence_count(text)
    cleaned_text = clean_text(text)
    prediction, confidence = predict_text(cleaned_text)
    print("model prediction:", prediction)
    polarity, subjectivity = get_sentiments(cleaned_text)
    count_per, count_loc, count_org = get_entities_features(cleaned_text)
    count_noun, count_verb, count_adj, count_adv = get_stylistic_features(cleaned_text)
    unique_words_ratio = get_unique_words_ratio(text)
    predicted_class = "Credible" if prediction == 0 else "Non-Credible"
    if confidence < 0.65:
        predicted_class = "Doubtful"
    return TextAnalysis(
        prediction=predicted_class,
        confidence=confidence,
        word_count=word_count,
        sent_count=sent_count,
        polarity=polarity,
        subjectivity=subjectivity,
        count_per=count_per,
        count_loc=count_loc,
        count_org=count_org,
        count_noun=count_noun,
        count_verb=count_verb,
        count_adj=count_adj,
        count_adv=count_adv,
        unique_words_ratio=unique_words_ratio,
        most_frequent_words=most_frequent_words
    )


def remove_special_chars(text: str) -> str:
    regex = '[^\w\s]'
    return re.sub(regex, '', text)


def remove_tags_and_lines(text: str) -> str:
    tags_re = "<[^>]+>"
    text = re.sub(tags_re, '', text)
    empty_line_re = "^\s*\n"
    text = re.sub(empty_line_re, '', text, flags=re.MULTILINE)
    return text.replace('\n', ' ')


def remove_urls(text: str) -> str:
    url_pattern = r'https?://\S+'
    return re.sub(url_pattern, '', text)


def remove_stopwords(text):
    arabic_stopwords = set(stopwords.words('arabic'))
    tokens = word_tokenize(text)
    filtered_tokens = [
        token for token in tokens if token not in arabic_stopwords]
    return ' '.join(filtered_tokens)


def stem(text: str) -> str:
    return stemmer.stem(text)


def clean_text(text: str):
    text = remove_urls(text)
    text = remove_special_chars(text)
    text = remove_tags_and_lines(text)
    text = araby.strip_tashkeel(text)
    text = araby.strip_tatweel(text)
    text = remove_stopwords(text)
    text = stem(text)
    text = segmenter.segment(text)
    return text


def predict_text(text: str) -> tuple[int, float]:
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
        predicted_class = int(probabilities.argmax(dim=1).item())
        credible_proba, non_credible_proba = probabilities[0].tolist()
        returned_proba = credible_proba if predicted_class == 0 else non_credible_proba
        return predicted_class, returned_proba


def get_entities_features(text: str) -> tuple:
    named_entity_recognizer = FarasaNamedEntityRecognizer()
    ner_text = named_entity_recognizer.recognize(text)
    count_per = 0
    count_loc = 0
    count_org = 0
    tokens = ner_text.split()
    for token in tokens:
        parts = token.rsplit('/', 1)
        if len(parts) == 2:
            tag = parts[1]
            if "PERS" in tag:
                count_per += 1
            elif "ORG" in tag:
                count_org += 1
            elif "LOC" in tag:
                count_loc += 1
    return count_per, count_loc, count_org


def get_stylistic_features(text: str) -> tuple:
    pos_tagger = FarasaPOSTagger()
    pos_tagged = pos_tagger.tag(text)
    pos_tokens = pos_tagged.split()
    noun_count = 0
    adj_count = 0
    adv_count = 0
    verb_count = 0
    for token in pos_tokens:
        parts = token.rsplit('/')
        if len(parts) == 2:
            tag = parts[1]
            if "NOUN" in tag:
                noun_count += 1
            if "ADJ" in tag:
                adj_count += 1
            if "V" == tag:
                verb_count += 1
            if "ADV" in tag:
                adv_count += 1
    return noun_count, verb_count, adj_count, adv_count


def get_word_and_sentence_count(text: str) -> tuple[int, int]:
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    return len(words), len(sentences)


def get_sentiments(text: str):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


def get_most_frequent_words(text: str):
    words = nltk.word_tokenize(text.lower())
    word_freq = {}
    for word in words:
        if word.isalnum():
            word_freq[word] = word_freq.get(word, 0) + 1
    _list = [{"keyword": word, "count": count}
             for word, count in word_freq.items()]
    print(word_freq)
    sorted_words = sorted(_list, key=lambda x: x["count"], reverse=True)
    return sorted_words[:3]


def get_unique_words_ratio(text: str):
    words = nltk.word_tokenize(text.lower())
    total_words = len(words)
    unique_words = len(set(words))
    return unique_words / total_words if total_words > 0 else 0


def cross_validate_news(text: str):

    prompt = f"""
                Search the web for information related to the following news article. Then, fact-check the article.
                Mention the sources names in the reason. Respond in JSON format, strictly adhering to the following structure, in English, without adding any extra text or explanations outside the JSON:
                {{
                "check": "Credible" or "Non-credible",
                "reason": "A concise explanation of the checking result, no more than three lines."
                }}
                News Article: {text}"
            """
    response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=GenerateContentConfig(
                tools=[google_search_tool],
                response_modalities=["TEXT"],
            )
        )
    try:
        json_string = re.sub("```", "", str(response.text))
        json_string = json_string.replace("json", "")
        json_string = json_string.strip()
        result = json.loads(json_string)
        print(result)
        return result
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        # print the response to debug.
        print(f"Response text: {response.text}")
        return None  # or return an error dictionary.
