import spacy

from collections import namedtuple
ParsedIntent = namedtuple('ParsedIntent', 'intent query topic')

class QueryIntent():

    def __init__(self):
        nlp = spacy.load("en_core_web_sm")
        intents = {"topic_count_top": "What is the hottest topic?", "person_answer_question_count_top": "Who answered the most questions", "question_answer_count_top": "Which question got most answers?", "person_answer_topic": "Which persons knows about x"}
        bag_of_words = {key: set([token.lemma_ for token in nlp(value)])for key, value in intents.items()}

        self.nlp = nlp
        self.intents = intents
        self.bag_of_words = bag_of_words

    def query_types(self):
        for t,q in self.intents.items():
            print(f"{q}\t(tag: {t})")
        print()

    def get_query_intent(self, query):

        # Returns None if no matching intent found otherwise it returns a ParsedIntent tuple

        extracted = self.nlp(query)
        parsed = set([token.lemma_ for token in extracted])

        counts = [(key, len(parsed.intersection(value))) for key, value in self.bag_of_words.items()]
        top_intents = sorted(counts, key=lambda tup: tup[1], reverse=True)

        top_intent = top_intents[0]
        if(top_intent[1] < 1):
            return None
        else:
            intent = top_intent[0]
            topic = None
            if intent == "person_answer_topic":
                topic = query.split("about")[-1]

            return ParsedIntent(intent=intent, query=query,topic=topic)

if __name__ == "__main__":
    
    query_intent = QueryIntent()

    print("Questions you can ask:")
    query_intent.query_types()
    
    query = input("Please input your question:")
    result = query_intent.get_query_intent(query)
    print("Intent:", result.intent, ",", "Topic", result.topic )
