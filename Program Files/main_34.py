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
import tkinter as tk
from tkinter import filedialog, messagebox

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfFileReader(file)
        num_pages = pdf_reader.numPages
        for page_num in range(num_pages):
            page = pdf_reader.getPage(page_num)
            text += page.extractText()
    return text

def extract_information_nlp(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return entities

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

def translate_text_nlp(text, source_lang, target_lang='en'):
    translator = Translator()
    translation = translator.translate(text, src=source_lang, dest=target_lang)
    return translation.text

def cross_lingual_analysis_nlp(text, source_lang='en', target_lang='fr'):
    if source_lang != 'en':
        translated_text = translate_text_nlp(text, source_lang, target_lang)
    else:
        translated_text = text
    return translated_text

def generate_summary_nlp(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count=3)
    return " ".join([str(sentence) for sentence in summary])

def process_pdf(pdf_path):
    pdf_text = extract_text_from_pdf(pdf_path)
    detected_language = detect(pdf_text)
    extracted_info = extract_information_nlp(pdf_text)
    topics = topic_modeling_nlp(extracted_info)
    documents = ["Document 1 content", "Document 2 content", "Document 3 content"]
    index_documents_nlp(documents)
    query = "query"
    search_results = search_index_nlp(query)
    analysis_result = cross_lingual_analysis_nlp(pdf_text, source_lang=detected_language, target_lang='en')
    summary = generate_summary_nlp(pdf_text)
    return extracted_info, topics, search_results, analysis_result, summary

def open_file():
    filename = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    if filename:
        try:
            extracted_info, topics, search_results, analysis_result, summary = process_pdf(filename)
            messagebox.showinfo("Analysis Results", f"Extracted Information: {extracted_info}\n\nTopics: {topics}\n\nSearch Results: {search_results}\n\nCross-lingual Analysis Result: {analysis_result}\n\nSummary: {summary}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

def main():
    root = tk.Tk()
    root.title("Legal Document Analysis")
    root.geometry("400x200")

    label = tk.Label(root, text="Legal Document Analysis", font=("Helvetica", 16))
    label.pack(pady=10)

    button = tk.Button(root, text="Open PDF", command=open_file)
    button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
