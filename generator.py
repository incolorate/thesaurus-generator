import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
import nltk
import re
import gc
from tqdm import tqdm

class ThesaurusGenerator:
    def __init__(self, similarity_threshold=0.75, batch_size=1000):
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.ps = PorterStemmer()
        
        # Download required NLTK data
        try:
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except:
            pass

    def preprocess_terms(self, terms):
        """Clean and standardize terms"""
        processed_terms = []
        for term in terms:
            if isinstance(term, str) and term.strip():  # Check if term is a non-empty string
                # Convert to lowercase and remove special characters except hyphen
                term = re.sub(r'[^a-zA-Z\s-]', '', term.lower())
                # Remove extra whitespace
                term = ' '.join(term.split())
                processed_terms.append(term)
        return processed_terms

    def find_similar_terms_batched(self, terms):
        """Find similar terms using TF-IDF and cosine similarity in batches"""
        if not terms:
            print("Warning: No terms provided to process")
            return []
            
        # Preprocess terms
        processed_terms = self.preprocess_terms(terms)
        if not processed_terms:
            print("Warning: No terms remained after preprocessing")
            return []
        
        # Create TF-IDF vectors for all terms
        print("Creating TF-IDF vectors...")
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
        tfidf_matrix = vectorizer.fit_transform(processed_terms)
        
        # Process in batches
        similar_groups = []
        used_terms = set()
        num_terms = len(terms)
        
        # Process in batches - outer loop for reference terms
        for i in tqdm(range(0, num_terms, self.batch_size), desc="Processing batches"):
            batch_end = min(i + self.batch_size, num_terms)
            batch_indices = list(range(i, batch_end))
            
            # Skip terms that are already used
            batch_indices = [idx for idx in batch_indices if idx not in used_terms]
            
            if not batch_indices:
                continue
                
            # Get batch vectors
            batch_vectors = tfidf_matrix[batch_indices]
            
            # Calculate similarities for this batch against all remaining terms
            remaining_indices = [j for j in range(num_terms) if j not in used_terms]
            
            if not remaining_indices:
                break
                
            remaining_vectors = tfidf_matrix[remaining_indices]
            
            # Calculate cosine similarity between this batch and remaining terms
            batch_similarities = cosine_similarity(batch_vectors, remaining_vectors)
            
            # Map indices back to original positions
            for batch_idx, orig_i in enumerate(batch_indices):
                if orig_i in used_terms:
                    continue
                    
                group = {orig_i}
                used_terms.add(orig_i)
                
                # Find similar terms above threshold
                for rem_idx, orig_j in enumerate(remaining_indices):
                    if orig_j > orig_i and orig_j not in used_terms:
                        if batch_similarities[batch_idx, rem_idx] > self.similarity_threshold:
                            group.add(orig_j)
                            used_terms.add(orig_j)
                
                if len(group) > 1:
                    similar_groups.append([terms[idx] for idx in group])
                    
            # Free memory
            del batch_vectors, remaining_vectors, batch_similarities
            gc.collect()
        
        return similar_groups

    def generate_thesaurus_file(self, keywords_string, output_file):
        """Generate VOSviewer thesaurus file from a comma-separated string of keywords"""
        # Split the keywords string into a list
        keywords = [k.strip() for k in keywords_string.split(',') if k.strip()]
        print(f"Processing {len(keywords)} keywords...")
        
        # Find similar terms with batching
        similar_groups = self.find_similar_terms_batched(keywords)
        print(f"Found {len(similar_groups)} groups of similar terms")
        
        # Generate thesaurus content
        thesaurus_content = []
        
        for group in similar_groups:
            # Use the shortest term as the preferred term
            preferred_term = min(group, key=len)
            for term in group:
                if term != preferred_term:
                    thesaurus_content.append(f"{term},{preferred_term}")
        
        # Sort entries alphabetically
        thesaurus_content.sort()
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in thesaurus_content:
                f.write(line + '\n')
        
        return len(thesaurus_content)

def main():
    # Read keywords from file in chunks to reduce memory usage
    keywords = []
    chunk_size = 10000  # Adjust based on your file size and available memory
    
    with open('test.txt', 'r') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            keywords.extend([k.strip() for k in chunk.replace('\n', ',').split(',') if k.strip()])
    
    # Filter out duplicates to reduce processing load
    keywords = list(dict.fromkeys(keywords))
    
    print(f"Loaded {len(keywords)} unique keywords")
    
    # Generate thesaurus file with batching
    generator = ThesaurusGenerator(similarity_threshold=0.75, batch_size=500)
    num_entries = generator.generate_thesaurus_file(','.join(keywords), 'vosviewer_thesaurus.txt')
    print(f"Generated thesaurus file with {num_entries} entries")

if __name__ == "__main__":
    main()