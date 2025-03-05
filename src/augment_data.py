# src/augment_data.py
import nlpaug.augmenter.word as naw

def augment_texts(texts, aug_factor=1):
    """
    Augments a list of texts using synonym replacement from WordNet.
    
    Args:
        texts (list of str): The input texts.
        aug_factor (int): Number of augmented versions to generate per text.
        
    Returns:
        list of str: A list containing the original and augmented texts.
    """
    augmenter = naw.SynonymAug(aug_src='wordnet')
    augmented_texts = []
    for text in texts:
        augmented_texts.append(text)  # Include the original text
        for _ in range(aug_factor):
            augmented_text = augmenter.augment(text)
            augmented_texts.append(augmented_text)
    return augmented_texts

if __name__ == "__main__":
    sample_texts = [
        "This movie was fantastic and full of surprises.",
        "I did not like the film; it was too long and boring."
    ]
    augmented = augment_texts(sample_texts, aug_factor=1)
    for idx, text in enumerate(augmented):
        print(f"{idx+1}: {text}")