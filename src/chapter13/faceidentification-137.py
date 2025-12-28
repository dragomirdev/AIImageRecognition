from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Assume emb1 and emb2 are 128-D embeddings from a model like FaceNet
emb1 = np.random.rand(128)
emb2 = np.random.rand(128)

# Compute distances
eu_dist = np.linalg.norm(emb1 - emb2)
cos_sim = cosine_similarity([emb1], [emb2])[0][0]

print(f"Euclidean Distance: {eu_dist:.4f}")
print(f"Cosine Similarity: {cos_sim:.4f}")

if eu_dist < 0.6:
    print("Faces likely belong to the same person.")
else:
    print("Faces likely belong to different people.")
