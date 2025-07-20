from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained SBERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Define the actual and generated answers
actual_answer = """
The Supreme Court decided that the allocation of the land to APPL was valid and that the permissions for conversion granted by RIICO were in accordance with the Master Plan and the 1979 Rules. The cancellation by the State Government and RIICO was set aside.
"""

generated_answer = """
The Supreme court held that the entire area (except for a small portion) was notified as an industrial area by the State Government under Section 17(6)(ii). Therefore it is liable to be treated as CLA. As per Rule 45C read with SRO-98 dated 05.04.2016, allotment of plots shall only be made after obtaining prior permission from competent authority.
"""

# Generate sentence embeddings for both answers
actual_embedding = model.encode(actual_answer, convert_to_tensor=False)
generated_embedding = model.encode(generated_answer, convert_to_tensor=False)

# Convert embeddings to 2D if needed
actual_embedding = actual_embedding.reshape(1, -1)
generated_embedding = generated_embedding.reshape(1, -1)

# Calculate cosine similarity
similarity_score = cosine_similarity(actual_embedding, generated_embedding)[0][0]

# Output similarity score
print(f"Similarity Score: {similarity_score}")