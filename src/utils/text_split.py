import re

def split_text_by_length_word_boundary(text, max_length):
    """Split text into chunks of specified maximum length, respecting word boundaries."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        # +1 for the space before the word (except for the first word)
        word_length = len(word) + (1 if current_length > 0 else 0)
        
        if current_length + word_length <= max_length:
            # Add to current chunk
            if current_length > 0:
                current_chunk.append(' ')
                current_length += 1
            current_chunk.append(word)
            current_length += len(word)
        else:
            # Start a new chunk
            chunks.append(''.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
    
    # Add the last chunk
    if current_chunk:
        chunks.append(''.join(current_chunk))
    
    return chunks


def split_text_by_sentences_and_length(text, max_length):
    """Split text into chunks, respecting sentence boundaries and maximum length."""
    # Split by sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        if current_length + len(sentence) + (1 if current_length > 0 else 0) <= max_length:
            # Add to current chunk
            if current_length > 0:
                current_chunk.append(' ')
                current_length += 1
            current_chunk.append(sentence)
            current_length += len(sentence)
        else:
            # If the sentence itself is longer than max_length, we need to split it
            if len(sentence) > max_length:
                # First add the current chunk if it's not empty
                if current_chunk:
                    chunks.append(''.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split the long sentence by words
                sentence_chunks = split_text_by_length_word_boundary(sentence, max_length)
                chunks.extend(sentence_chunks[:-1])
                
                # Add the last part to the current chunk
                current_chunk = [sentence_chunks[-1]]
                current_length = len(sentence_chunks[-1])
            else:
                # Start a new chunk with this sentence
                chunks.append(''.join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
    
    # Add the last chunk
    if current_chunk:
        chunks.append(''.join(current_chunk))
    
    return chunks