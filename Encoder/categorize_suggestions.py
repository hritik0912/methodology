import json
import re
import os

# NEW: A set of common English words to filter out from our Hinglish vocabulary.
# This helps prevent purely English sentences from being misclassified.
ENGLISH_STOP_WORDS = set([
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'did', 'do',
    'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having',
    'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it',
    'its', 'itself', 'just', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'of', 'off', 'on',
    'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 's', 'same', 'she', 'should',
    'so', 'some', 'such', 't', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there',
    'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were',
    'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'automation', 'monitoring', 'public', 'services', 'education', 'stream',
    'theory', 'practicals', 'school', 'colleges', 'boards', 'amendments', 'change', 'country', 'generations',
    'students', 'related', 'pressure', 'financial', 'emotional', 'empathy', 'sympathy', 'sessions', 'based',
    'case', 'facts', 'day', 'life', 'freedom', 'express', 'execute', 'teachers', 'faculty', 'trained',
    'feedback', 'guide', 'situation', 'ideas', 'culture', 'careers', 'sincerely', 'digital', 'ecosystem',
    'platform', 'policy', 'mandates', 'integration', 'real-time', 'assessment'
])

def build_hinglish_vocab(filepath):
    """
    Reads the cleaned_data.json file and creates a set of unique Hinglish words,
    MINUS common English words.
    """
    print("Building Hinglish vocabulary...")
    hinglish_words = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                words = re.findall(r'\b\w+\b', item.get('hinglish', '').lower())
                hinglish_words.update(words)
        
        # --- MODIFIED PART ---
        # Refine the vocabulary by removing common English stop words.
        refined_vocab = hinglish_words - ENGLISH_STOP_WORDS
        print(f"Built initial vocabulary with {len(hinglish_words)} words.")
        print(f"Refined vocabulary to {len(refined_vocab)} unique non-stop Hinglish words.")
        return refined_vocab
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}.")
        return None

def is_hindi(text):
    """
    Checks if the text contains a significant number of Devanagari characters.
    """
    hindi_chars = 0
    total_chars = 0
    for char in text:
        if '\u0900' <= char <= '\u097F':
            hindi_chars += 1
        if char.isprintable() and not char.isspace():
            total_chars += 1
    return total_chars > 0 and (hindi_chars / total_chars) > 0.1

def is_hinglish(text, hinglish_vocab):
    """
    Checks if a high percentage of words in the text are in the refined Hinglish vocabulary.
    """
    if not hinglish_vocab:
        return False
        
    words = set(re.findall(r'\b\w+\b', text.lower()))
    if not words:
        return False
    
    common_words = words.intersection(hinglish_vocab)
    
    # --- MODIFIED PART ---
    # Lowered the threshold because we are matching on a smaller, more specific set of words.
    return (len(common_words) / len(words)) > 0.3

def categorize_suggestions(input_filepath, hinglish_vocab):
    """
    Reads the suggestions file and categorizes each entry.
    """
    print(f"\nReading and categorizing suggestions from {input_filepath}...")
    
    categories = {"hindi": [], "english": [], "hinglish": []}
    
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            suggestions = json.load(f)
    except FileNotFoundError:
        print(f"Error: The input file {input_filepath} was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_filepath}.")
        return

    for item in suggestions:
        suggestion_text = item.get("SuggestionText", "")
        
        if not suggestion_text:
            continue

        if is_hindi(suggestion_text):
            categories["hindi"].append(item)
        elif is_hinglish(suggestion_text, hinglish_vocab):
            categories["hinglish"].append(item)
        else:
            categories["english"].append(item)
            
    print("Categorization complete.")
    return categories

def save_categorized_data(categories, output_dir):
    """
    Saves the categorized data into separate JSON files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    for lang, data in categories.items():
        output_filepath = os.path.join(output_dir, f"{lang}_suggestions.json")
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(data)} suggestions to {output_filepath}")

def main():
    hinglish_data_path = 'Hinglish2Hindi/cleaned_data.json'
    suggestions_path = 'translate/Suggestion1Lk.json'
    output_directory = 'categorized_suggestions'

    hinglish_vocab = build_hinglish_vocab(hinglish_data_path)
    
    if hinglish_vocab is None:
        print("Halting execution due to vocabulary building failure.")
        return

    categorized_data = categorize_suggestions(suggestions_path, hinglish_vocab)

    if categorized_data:
        save_categorized_data(categorized_data, output_directory)
        print("\nProcess finished successfully!")

if __name__ == "__main__":
    main()