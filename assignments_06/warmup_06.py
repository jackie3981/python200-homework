import os
import string
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

BASE_DIR = os.path.dirname(__file__)
pdf_path = os.path.join(BASE_DIR, "brightleaf_pdfs")

docs = SimpleDirectoryReader(pdf_path).load_data()

# ** RAG Concepts **
def concepts_Q1():
    """
    Scenario A: A legal team wants an assistant that can answer questions about their internal policy library — hundreds of PDFs that are updated every quarter.
    Best Approach: RAG
    Reasoning: The legal team has hundreds of PDFs that change every quarter, so the model needs access to constantly updated information.
    RAG is best because it retrieves relevant documents at query time without needing to retrain the model every time policies change.

    Scenario B: A startup wants their model to write product copy in a very specific brand voice — a dry, minimalist style that does not appear much online. 
    They have 3,000 examples their in-house writers produced over the years.
    Best Approach: Fine-tuning
    Reasoning: The startup wants a very specific writing style that is unique to their brand. Since they already have 3,000 examples of past product copy, fine-tuning is a good
    choice because it teaches the model consistent tone, structure, and voice.

    Scenario C: A data analyst needs to ask an LLM questions about a single two-page report she just received. She does not need this to work for any other document.
    Best Approach: Prompt Engineering
    Reasoning: The analyst only needs to work with a single short report one time. The simplest solution is to provide the report directly in the prompt and ask questions about it,
    without building a RAG pipeline or fine-tuning a model.
    """
    pass

def concepts_Q2():
    """
    AI Hallucinations

    A confidently wrong answer is more harmful than a response that says "I am not sure" because people are more likely to trust and act on information
    that sounds certain and authoritative. When a model expresses doubt, users are more likely to verify the information themselves.

    For example, a medical AI assistant that confidently gives incorrect dosage instructions for a medication could seriously harm a patient. If the AI instead
    said it was uncertain, the user would be more likely to consult a doctor or pharmacist before taking action.

    The tone of the response matters because confidence affects trust. We often associate clear and confident language with expertise, even when the information
    is incorrect. A calm and authoritative tone can make hallucinated information seem believable, which increases the risk that users will follow bad advice.
    """
    pass

def concepts_Q3():
    """
    Initial order of steps:
    steps = [
        "Generate a response from the LLM",
        "Extract text from source documents",
        "Receive the user's query",
        "Retrieve the most relevant chunks",
        "Convert text chunks into embeddings",
        "Inject retrieved chunks into the prompt",
        "Split text into chunks",
        "Embed the user's query",
    ]

    Correct order of steps:
        1. "Extract text from source documents"
           The system reads and extracts text from PDFs, files, or other data sources.
        2. "Split text into chunks"
           Large documents are divided into smaller sections so they can be searched efficiently.
        3. "Convert text chunks into embeddings"
           Each chunk is transformed into a numerical vector representation and stored in a vector database.
        4. "Receive the user's query"
           The user asks a question or submits a request.
        5. "Embed the user's query"
           The user's query is converted into an embedding vector.
        6. "Retrieve the most relevant chunks"
           The system compares the query embedding with stored embeddings to find the most relevant text chunks.
        7. "Inject retrieved chunks into the prompt"
           The retrieved information is added to the prompt sent to the language model.
        8. "Generate a response from the LLM"
           The language model generates a final answer using both the query and the retrieved context.

    """
    pass

# ** Keyword RAG **
def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]

def keyword_selected_document(query):
   documents = {
      "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
      "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
      "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
      "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
   }

   result = simple_keyword_retrieval(query,documents)
   print(f"Document selected: {result}")

# ** Semantic RAG Concepts **
# Semantic Question 1
# A: What is a vector embedding? (1-2 sentences)
# Q: A vector embedding is a way of turning text into a list of numbers that represents its meaning in a high-dimensional space. 
#    Texts with similar meanings end up with similar vectors, even if they use different words.

# Q: Two text chunks have cosine similarity scores of 0.85 and 0.30 with a given query. Which chunk is more relevant, 
#    and what does that number tell you about the relationship between the texts?
# A: The chunk with cosine similarity 0.85 is more relevant because it is closer in meaning to the query. The score of 0.30 indicates low semantic similarity, 
#    meaning the second chunk is only weakly related to the query in meaning, even if it may share some words.

# Q: Why can semantic search find a relevant chunk even when none of the exact words from the query appear in the chunk?
# A: Semantic search works because embeddings capture meaning instead of exact words. This allows the system to match concepts 
#    even if the same words are not used in the text.

# Semantic Question 2
"""
| Feature                    | Keyword RAG                       | Semantic RAG                      |
|----------------------------|-----------------------------------|-----------------------------------|
| What is compared?          | Exact word overlap                | Vector similarity                 |
| What is retrieved?         | Full document                     | Relevant text chunks              |
| Can it handle synonyms?    | No                                | Yes                               |
| Storage format             | Plain text dictionary             | Vector database                   |
| Relevance score            | Number of overlapping keywords    | Cosine similarity between vectors |
"""

# ** LlamaIndex **
def llamaindex_pipeline(questions, similarity=3):
   # Load documents directly from PDFs in the folder
   docs = SimpleDirectoryReader(pdf_path).load_data()
   # Build a vector index automatically (handles chunking + embeddings)
   index = VectorStoreIndex.from_documents(docs)
   query_engine = index.as_query_engine(similarity_top_k=similarity)
   for q in questions:
      print(f"\nQuestion: {q}")
      response = query_engine.query(q)
      print("Answer:", response)
      
      for node_with_score in response.source_nodes:
         print(f"Node ID: {node_with_score.node.node_id}")
         print(f"Similarity Score: {node_with_score.score:.4f}")
         print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
         print("-" * 60)
   return query_engine

def llamaindex_Q4(query_engine):
    llm = LlamaOpenAI(model="gpt-4o-mini", temperature=0.2)
    faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
    relevancy_evaluator = RelevancyEvaluator(llm=llm)

    # Query 1: expected good response
    query1 = "What employee benefits does BrightLeaf offer?"
    response1 = query_engine.query(query1)
    print(f"Response 1: {response1}")
    faith1 = faithfulness_evaluator.evaluate_response(query=query1, response=response1)
    relev1 = relevancy_evaluator.evaluate_response(query=query1, response=response1)
    print(f"\nQuery: {query1}")
    print(f"Faithfulness Score: {faith1.score}")
    print(f"Relevancy Score: {relev1.score}")

    # Query 2: expected lower-quality response
    query2 = "What is BrightLeaf's stock price?"
    response2 = query_engine.query(query2)
    print(f"Response 2: {response2}")
    faith2 = faithfulness_evaluator.evaluate_response(query=query2, response=response2)
    relev2 = relevancy_evaluator.evaluate_response(query=query2, response=response2)
    print(f"\nQuery: {query2}")
    print(f"Faithfulness Score: {faith2.score}")
    print(f"Relevancy Score: {relev2.score}")



if load_dotenv():
   print("API key loaded successfully.")
   # RAG Concepts
   #RAG_Q1
   concepts_Q1()
   #RAG_Q2
   concepts_Q2()
   #RAG_Q3
   concepts_Q3()

   # Keyword RAG
   # Keyword_Q1
   query_q1 = "What are your hours on the weekend?"
   keyword_selected_document(query_q1)
   # The system selects documents based on exact keyword matches with the query. Both "hiring.txt" and "loyalty.txt" matched the word "your", but since
   # they had the same score, "loyalty.txt" was selected after sorting. hours.txt had a score of 0 because the exact word "hours" does not appear in the document,
   # and "weekend" (from the query) does not match "weekends" (in the document) since keyword RAG requires exact word matches. 

   # Keyword_Q2
   query_q2 = "Do you have anything without caffeine?"
   keyword_selected_document(query_q2)
   # Q: Which document was selected
   # A: No document was selected because there were no exact keyword matches between the query and the documents.

   # Q: Whether keyword RAG got this right — and why or why not
   # A: Keyword RAG did not work well here because it only searches for exact word overlap. Even though "menu.txt" is related to drinks and caffeine, 
   #    it does not contain the exact word "caffeine".

   # Q: What kind of retrieval would do better here
   # A: Semantic retrieval using embeddings would perform better because it can understand related meanings and concepts.

   # Keyword_Q3
   query_q3 = "How do I sign up for rewards?"
   keyword_selected_document(query_q3)
   # My prediction was correct. I expected no matches because ['do', 'how', 'i', 'rewards', 'sign', 'up'] doesn't appear in any of the documents after cleaning.
   # The result confirmed that all documents had a score of 0 due to lack of exact keyword overlap. 

   # LlamaIndex
   # LlamaIndex_Q1
   questions = [
      "What employee benefits does BrightLeaf offer?",
      "What are BrightLeaf's security policies?",
   ]

   query_engine = llamaindex_pipeline(questions)
   # Query 1: "What employee benefits does BrightLeaf offer?"
   # The retrieved chunks are relevant — the first one (score 0.9106) comes directly from the employee benefits document.
   # The response sounds confident and specific, listing concrete benefits without any hedging phrases.
   # Unexpectedly, the third chunk came from the security policies document, which is unrelated to benefits.

   # Query 2: "What are BrightLeaf's security policies?"
   # The retrieved chunks are relevant — the first one (score 0.8824) comes directly from the security policy document.
   # The response sounds confident and detailed, listing specific security measures without any hedging phrases.
   # Unexpectedly, the second chunk came from the employee benefits document, which is unrelated to security policies.

   # LlamaIndex_Q2
   similarity_k1 = 1
   similarity_k5 = 5
   print("******* LlamaIndex_Q2 k=1 *******")
   llamaindex_pipeline(questions, similarity_k1)
   print("******* LlamaIndex_Q2 k=5 *******")
   llamaindex_pipeline(questions, similarity_k5)
   # With k=1, the response is accurate but less detailed, as only the most relevant chunk is used.
   # With k=5, the response is more detailed because more context is available, but irrelevant chunks (partnerships, earnings report) are also retrieved, 
   # which adds noise without improving the answer.
   # More context is not always better — irrelevant chunks increase token usage and can confuse the model.

   # LlamaIndex_Q3
   query_q3 = "How do BrightLeaf's financial results relate to their partnerships?"
   llamaindex_pipeline([query_q3])
   # Query: "How do BrightLeaf's financial results relate to their partnerships?"
   # The retrieved chunks are relevant — the first comes from partnerships and the second from the earnings report, which is exactly what this cross-document query requires.
   # However, the response uses phrases like "likely played a role", which indicates the model is inferring a connection that is not explicitly stated in the documents. 
   # This is a subtle form of hallucination, the retrieval worked correctly but the model went beyond what the context actually says.
   # To improve this, the prompt could instruct the model to only state what is explicitly in the documents and avoid drawing inferences.

   # LlamaIndex_Q4
   llamaindex_Q4(query_engine)
   # Q: What does a faithfulness score of 1.0 mean? What would a score of 0.0 indicate?
   # A: Faithfulness score of 1.0 means the response is fully grounded in the retrieved context — no hallucinations.
   #    A score of 0.0 would indicate the response contains information not supported by the retrieved chunks.

   # Q: What does a relevancy score measure, and how is it different from faithfulness?
   # A: Relevancy measures whether the response addresses the query given the retrieved context.
   #    It differs from faithfulness: faithfulness checks if the response is grounded in the context,
   #    while relevancy checks if the response actually answers the question.

   # Q: Did the scores change between your two queries? If so, why do you think that happened?
   # A: Both queries scored 1.0. The second query ("What is BrightLeaf's stock price?") also scored 1.0, because the model correctly responded "this information is not in the context", 
   #    which is both faithful (no hallucination) and relevant (it addressed the query appropriately).
   #    A lower score would appear if the model had hallucinated a stock price instead.

   # Q: What is the "LLM-as-a-judge" approach, and why is it used for RAG evaluation instead of a simple accuracy metric?
   # A: The "LLM-as-a-judge" approach uses a separate LLM to evaluate the quality of responses
   #    because RAG output is in natural language, there is no single correct answer to compare against.
   #    Traditional accuracy metrics require an exact match, which does not work for open-ended text.
   #    An LLM judge can assess meaning, faithfulness, and relevance in the same way a human expert would.

else:
    print("Warning: could not load API key. Check your .env file.")