from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

docs_dir = Path("assignments_06/groundwork_docs")
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]
step_q5 = ["Do you have a parking lot?"]

# Step 1: Setup
def load_dir():
    assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

# Step 2: Load the Documents
def load_documents():
    docs = SimpleDirectoryReader(docs_dir).load_data()
    print("Documents loaded: ", len(docs))
    print("Filenames:")
    for i, doc in enumerate(docs):
        print(f"{i+1}. {doc.metadata['file_name']}")
    return docs

# Step 3: Build the Index and Query Engine
def build_index_queryengine(docs, similarity=3):
    index = VectorStoreIndex.from_documents(docs)
    try:
        query_engine = index.as_query_engine(similarity_top_k=similarity)
        print("Index built successfully. Ready to answer questions.")
    except Exception as e:
        print(f"Error initializing query engine: {e}")
    return query_engine

# Step 4: Query the Assistant
def query_assistant(query_engine, questions=questions):
    for q in questions:
        print(f"\nQuestion: {q}")
        response = query_engine.query(q)
        print("Answer:", response)
        for node_with_score in response.source_nodes:
            print(f"Node ID: {node_with_score.node.node_id}")
            print(f"Similarity Score: {node_with_score.score:.4f}")
            print(f"Text Snippet: {node_with_score.node.get_content()[:200]}...")
            print("-" * 60)
    
# Step 5: Find a Failure


if load_dotenv():
    print("API key loaded successfully.")

    print("Step 1: Setup")
    load_dir()

    print("Step 2: Load the Documents")
    docs = load_documents()

    print("Step 3: Build the Index and Query Engine")
    q_engine = build_index_queryengine(docs)

    print("Step 4: Query the Assistant")
    query_assistant(q_engine)
    # Q: After running all five queries, add a comment reflecting on the responses: did the assistant sound confident and accurate?
    #    Did any of the answers surprise you?
    # A: Overall, the assistant sounded confident and accurate across most questions.
    #    Details regarding hours, the founding story, and catering/wholesale were answered correctly, and the retrieved chunks were clearly relevant.
    #    The loyalty program response was also surprisingly detailed and complete. Even though the top retrieved chunk was the FAQ hours section 
    #    rather than a loyalty-specific document, the model successfully found the correct information.
    #    The dairy-free question was the weakest point. The answer only stated that options are available at no extra charge, 
    #    but failed to list the actual milk alternatives (e.g., oat, almond). 
    #    This suggests that the relevant information may be spread across chunks in a way that the top-three retrieval did not fully capture.

    print("Step 5: Find a Failure")
    query_assistant(q_engine, step_q5)
    # Query: "Do you have a parking lot?"
    # This information is currently missing from the documentation, and the retrieval system pulled irrelevant chunks from the FAQ, Wholesale, 
    # and Our Story sections with low similarity scores (0.71-0.73).
    # The model handled this correctly by admitting the information was unavailable rather than hallucinating an answer. 
    # It also maintained an appropriately uncertain tone, which is ideal for a production RAG system.
    # To improve the user experience:
    # - Add an FAQ entry regarding parking, street access, or public transport.
    # - Implement a fallback response such as, "For questions not covered in our documents, please contact us at ..."

else:
    print("Warning: could not load API key. Check your .env file.")

#1. The lesson built semantic RAG manually — chunking, embedding, and indexing took many lines of code. 
#   How many lines did the equivalent LlamaIndex implementation take in your project? 
#   What does that tell you about the value of using a framework?

#   Using LlamaIndex reduced our core implementation to roughly four lines of code by utilizing SimpleDirectoryReader, VectorStoreIndex, 
#   and the query engine. In contrast, the manual semantic RAG approach required dozens of lines to separately manage chunking, 
#   embedding, indexing, and retrieval.
#   While frameworks are highly efficient for handling well-tested logic, they can sometimes obscure the underlying processes. 
#   Building the system manually first was a valuable exercise for understanding the fundamentals before transitioning to a more streamlined framework.

#2. You have now built a system that answers questions from real documents. 
#   Describe a different use case — not a coffee shop — where this approach would 
#   add genuine value to a business or organization.

#   An insurance sales company could utilize this approach to help customers navigate policy options and answer questions 
#   regarding coverage details, premiums, exclusions, and claims procedures.
#   By using this method, documents such as policy terms, FAQs, and pricing guides can be updated as products change 
#   without the need to retrain a model. This reduces the workload on agents handling repetitive inquiries 
#   while ensuring customers receive accurate and consistent information.

#3. What is one failure mode that RAG cannot fully prevent, even when retrieval is working correctly?

#   RAG has limitations in preventing hallucinations during the generation step.
#   Even when retrieval is successful and the correct context is provided, the LLM may still produce answers that exceed 
#   the provided information, sometimes adding details or figures not found in the source documents. 
#   While RAG effectively manages the information available to the model, it cannot fully control how the 
#   model interprets or generates text from that data.