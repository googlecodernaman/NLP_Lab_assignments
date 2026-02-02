"""
NLP Assignment: Tokenization, Stemming, and Lemmatization using NLTK
This script demonstrates:
1. Various tokenization techniques (Whitespace, Punctuation-based, Treebank, Tweet, MWE)
2. Stemming using Porter Stemmer and Snowball Stemmer
3. Lemmatization using WordNet Lemmatizer
"""

import nltk
from nltk.tokenize import (
    WhitespaceTokenizer,
    WordPunctTokenizer,
    TreebankWordTokenizer,
    TweetTokenizer,
    MWETokenizer
)
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer

# Download NLTK data
print("Downloading required NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
print("Download complete!\n")

# Sample texts 
sample_text = "Hello! This is a sample text for tokenization. It contains multiple sentences, punctuation marks, and words like running, jumps, better, wolves."
tweet_text = "Just loving this #NLP tutorial! 😊 Check it out @ https://example.com @username"
mwe_text = "New York is a great city. I love New York and San Francisco."

print("="*80)
print("TOKENIZATION TECHNIQUES")
print("="*80)

# 1. Whitespace Tokenization
print("\n1. WHITESPACE TOKENIZATION")
print("-" * 50)
print(f"Original Text: {sample_text}")
whitespace_tokenizer = WhitespaceTokenizer()
whitespace_tokens = whitespace_tokenizer.tokenize(sample_text)
print(f"Tokens: {whitespace_tokens}")
print(f"Number of tokens: {len(whitespace_tokens)}")

# 2. Punctuation-based Tokenization
print("\n2. PUNCTUATION-BASED TOKENIZATION (WordPunctTokenizer)")
print("-" * 50)
print(f"Original Text: {sample_text}")
wordpunct_tokenizer = WordPunctTokenizer()
wordpunct_tokens = wordpunct_tokenizer.tokenize(sample_text)
print(f"Tokens: {wordpunct_tokens}")
print(f"Number of tokens: {len(wordpunct_tokens)}")

# 3. Treebank Tokenization
print("\n3. TREEBANK TOKENIZATION")
print("-" * 50)
print(f"Original Text: {sample_text}")
treebank_tokenizer = TreebankWordTokenizer()
treebank_tokens = treebank_tokenizer.tokenize(sample_text)
print(f"Tokens: {treebank_tokens}")
print(f"Number of tokens: {len(treebank_tokens)}")

# 4. Tweet Tokenization
print("\n4. TWEET TOKENIZATION")
print("-" * 50)
print(f"Original Text: {tweet_text}")
tweet_tokenizer = TweetTokenizer()
tweet_tokens = tweet_tokenizer.tokenize(tweet_text)
print(f"Tokens: {tweet_tokens}")
print(f"Number of tokens: {len(tweet_tokens)}")

# 5. Multi-Word Expression (MWE) Tokenization
print("\n5. MULTI-WORD EXPRESSION (MWE) TOKENIZATION")
print("-" * 50)
print(f"Original Text: {mwe_text}")
# First tokenize normally
base_tokens = wordpunct_tokenizer.tokenize(mwe_text)
print(f"Base Tokens: {base_tokens}")

# multi-word expressions
mwe_tokenizer = MWETokenizer([('New', 'York'), ('San', 'Francisco')])
mwe_tokens = mwe_tokenizer.tokenize(base_tokens)
print(f"MWE Tokens: {mwe_tokens}")
print(f"Notice how 'New York' and 'San Francisco' are treated as single tokens")

print("\n" + "="*80)
print("STEMMING TECHNIQUES")
print("="*80)

# Sample words for stemming
words_for_stemming = ['running', 'runs', 'ran', 'runner', 'easily', 'fairly', 
                      'jumps', 'jumping', 'jumped', 'better', 'wolves', 
                      'organization', 'organized', 'organizing']

# 6. Porter Stemmer
print("\n6. PORTER STEMMER")
print("-" * 50)
porter = PorterStemmer()
print(f"{'Original Word':<20} {'Stemmed Word':<20}")
print("-" * 40)
for word in words_for_stemming:
    stemmed = porter.stem(word)
    print(f"{word:<20} {stemmed:<20}")

# 7. Snowball Stemmer (English)
print("\n7. SNOWBALL STEMMER (English)")
print("-" * 50)
snowball = SnowballStemmer('english')
print(f"{'Original Word':<20} {'Stemmed Word':<20}")
print("-" * 40)
for word in words_for_stemming:
    stemmed = snowball.stem(word)
    print(f"{word:<20} {stemmed:<20}")

# Comparison
print("\n8. COMPARISON: PORTER vs SNOWBALL STEMMER")
print("-" * 50)
print(f"{'Original Word':<20} {'Porter':<20} {'Snowball':<20}")
print("-" * 60)
for word in words_for_stemming:
    porter_stem = porter.stem(word)
    snowball_stem = snowball.stem(word)
    print(f"{word:<20} {porter_stem:<20} {snowball_stem:<20}")

print("\n" + "="*80)
print("LEMMATIZATION")
print("="*80)

# 9. WordNet Lemmatizer
print("\n9. WORDNET LEMMATIZER")
print("-" * 50)
lemmatizer = WordNetLemmatizer()

words_with_pos = [
    ('running', 'v'),  
    ('runs', 'v'),     
    ('ran', 'v'),      
    ('better', 'a'),   
    ('wolves', 'n'),   
    ('cacti', 'n'),    
    ('geese', 'n'),    
    ('organization', 'n'),  
]

print(f"{'Original Word':<20} {'POS':<10} {'Lemmatized':<20}")
print("-" * 50)
for word, pos in words_with_pos:
    lemma = lemmatizer.lemmatize(word, pos=pos)
    print(f"{word:<20} {pos:<10} {lemma:<20}")

# Default lemmatization
print("\n10. LEMMATIZATION WITHOUT POS TAG (defaults to noun)")
print("-" * 50)
print(f"{'Original Word':<20} {'Lemmatized (default)':<20}")
print("-" * 40)
for word in words_for_stemming:
    lemma = lemmatizer.lemmatize(word)
    print(f"{word:<20} {lemma:<20}")

print("\n" + "="*80)
print("COMPLETE COMPARISON: STEMMING vs LEMMATIZATION")
print("="*80)
print(f"{'Word':<15} {'Porter':<15} {'Snowball':<15} {'Lemmatizer':<15}")
print("-" * 60)
for word in words_for_stemming:
    porter_stem = porter.stem(word)
    snowball_stem = snowball.stem(word)
    lemma = lemmatizer.lemmatize(word, pos='v')
    print(f"{word:<15} {porter_stem:<15} {snowball_stem:<15} {lemma:<15}")

print("\n" + "="*80)
print("PRACTICAL EXAMPLE: Processing a Full Sentence")
print("="*80)

sentence = "The striped bats are hanging on their feet for best results"
print(f"\nOriginal Sentence: {sentence}")

# Tokenize
tokens = treebank_tokenizer.tokenize(sentence)
print(f"\nTokens: {tokens}")

# Stem with Porter
print("\nPorter Stemming:")
porter_stems = [porter.stem(token) for token in tokens]
print(f"Stems: {porter_stems}")
print(f"Reconstructed: {' '.join(porter_stems)}")

# Stem with Snowball
print("\nSnowball Stemming:")
snowball_stems = [snowball.stem(token) for token in tokens]
print(f"Stems: {snowball_stems}")
print(f"Reconstructed: {' '.join(snowball_stems)}")

# Lemmatize
print("\nLemmatization (assuming verbs):")
lemmas = [lemmatizer.lemmatize(token, pos='v') for token in tokens]
print(f"Lemmas: {lemmas}")
print(f"Reconstructed: {' '.join(lemmas)}")

print("\n" + "="*80)
print("Analysis Complete!")
print("="*80)
