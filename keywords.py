from keybert import KeyBERT
import spacy
import statistics

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


class KeywordExtractor:
    def __init__(self):
        self.kw_model = KeyBERT()

        nlp = spacy.load("en_core_web_sm")

        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "tagger"]
        nlp.disable_pipes(*other_pipes)

        self.nlp = nlp

    def ranker(self, question, answers):

        question_keywords = self.extract(question)
        words, scores = zip(*question_keywords)

        results = []
        for index, answer in enumerate(answers):
            akeywords = self.extract(answer, candidates=words)
            awords, ascores = zip(*akeywords)
     
            results.append((index, answer, statistics.mean(ascores)))

        results = list(sorted(results, key=lambda tup: tup[2], reverse=True))
        return results

    def _keywords(self, text, candidates=None):
        return self.kw_model.extract_keywords(
            text,
            stop_words="english",
            keyphrase_ngram_range=(1, 2),
            candidates=candidates,
        )
    

    def extract(self, text, candidates=None):

        keywords = self._keywords(text, candidates=candidates)

        words, scores = zip(*keywords)

        filtered_keywords = []
        for index, doc in enumerate(self.nlp.pipe(words)):
            if len(doc) > 0 and "N" in doc[0].tag_:
                filtered_keywords.append(keywords[index])

        filtered_keywords = sorted(
            filtered_keywords, key=lambda tup: tup[1], reverse=True
        )
        return filtered_keywords

    def cluster_keywords(self, keywords, n=2, min_len=4):
        candidate_embeddings = self.kw_model.model.embed(keywords)
        kmeans = KMeans(n_clusters=n, random_state=0).fit(candidate_embeddings)

        lookup = defaultdict(list)
        topics = dict()
        for i, label in enumerate(kmeans.labels_):
            lookup[label].append(i)

        for label, center in enumerate(kmeans.cluster_centers_):
            scores = []
            for i in lookup[label]:
                similarity_score = cosine_similarity(
                    center.reshape(1, -1), candidate_embeddings[i].reshape(1, -1)
                )
                # Discard short keywords
                if len(keywords[i]) >= min_len:
                    scores.append((keywords[i], similarity_score))

            topic = sorted(scores, key=lambda tup: tup[1], reverse=True)[0][0]
            topics[label] = topic

        return list(map(lambda x: topics[x], kmeans.labels_))
