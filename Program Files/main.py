import PyPDF2
import spacy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from gensim.models import LdaModel
from gensim import corpora
from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser
from googletrans import Translator
from langdetect import detect

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfFileReader(file)
        num_pages = pdf_reader.numPages
        for page_num in range(num_pages):
            page = pdf_reader.getPage(page_num)
            text += page.extractText()
    return text

# Function to extract information using NLP
def extract_information_nlp(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return entities

# Function for topic modeling using NLP
def topic_modeling_nlp(texts):
    nlp = spacy.load("en_core_web_sm")
    texts_processed = []
    for text in texts:
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        texts_processed.append(tokens)
    dictionary = corpora.Dictionary(texts_processed)
    corpus = [dictionary.doc2bow(text) for text in texts_processed]
    lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary)
    return lda_model.print_topics()

# Function for legal information retrieval using NLP
def index_documents_nlp(documents):
    schema = Schema(content=TEXT(stored=True))
    ix = create_in("indexdir", schema)
    writer = ix.writer()
    for doc in documents:
        writer.add_document(content=doc)
    writer.commit()

def search_index_nlp(query):
    ix = open_dir("indexdir")
    searcher = ix.searcher()
    query_parser = QueryParser("content", ix.schema)
    query = query_parser.parse(query)
    results = searcher.search(query, limit=None)
    return [hit['content'] for hit in results]

# Function for language translation using NLP
def translate_text_nlp(text, source_lang, target_lang='en'):
    translator = Translator()
    translation = translator.translate(text, src=source_lang, dest=target_lang)
    return translation.text

# Function for cross-lingual analysis using NLP
def cross_lingual_analysis_nlp(text, source_lang='en', target_lang='fr'):
    if source_lang != 'en':
        translated_text = translate_text_nlp(text, source_lang, target_lang)
    else:
        translated_text = text
    # Perform analysis on translated text here
    return translated_text

# Function for text summarization using NLP
def generate_summary_nlp(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count=3) # Adjust the number of sentences as needed
    return " ".join([str(sentence) for sentence in summary])

# Example Usage
pdf_path = "your_legal_document.pdf"
pdf_text = extract_text_from_pdf(pdf_path)

# Detect language of the extracted text
detected_language = detect(pdf_text)

# Information Extraction using NLP
extracted_info = extract_information_nlp(pdf_text)
print("Extracted Information:", extracted_info)

# Topic Modeling using NLP
topics = topic_modeling_nlp(extracted_info)
print("Topics:", topics)

# Legal Information Retrieval using NLP
documents = ["Document 1 content", "Document 2 content", "Document 3 content"]
index_documents_nlp(documents)
query = "query"
search_results = search_index_nlp(query)
print("Search Results:", search_results)

# Cross-lingual Analysis using NLP
analysis_result = cross_lingual_analysis_nlp(pdf_text, source_lang=detected_language, target_lang='en')
print("Cross-lingual Analysis Result:", analysis_result)

# Summarization using NLP
summary = generate_summary_nlp(pdf_text)
print("Summary:", summary)
