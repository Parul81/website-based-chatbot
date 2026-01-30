"""
Text Chunking Module
Splits large text into smaller, meaningful chunks with metadata.
"""

from typing import List, Dict
import re


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100, 
               url: str = "", title: str = "") -> List[Dict]:
    """
    Split text into chunks with overlap and metadata.
    
    Args:
        text: The text to chunk
        chunk_size: Maximum characters per chunk
        chunk_overlap: Number of characters to overlap between chunks
        url: Source URL for metadata
        title: Page title for metadata
        
    Returns:
        List of dictionaries, each containing 'text' and 'metadata'
    """
    if not text:
        return []
    
    chunks = []
    
    # Split by sentences first for better chunking
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    current_chunk = ""
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        sentence_length = len(sentence)
        
        # If adding this sentence would exceed chunk_size
        if current_length + sentence_length > chunk_size and current_chunk:
            # Save current chunk
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': {
                    'url': url,
                    'title': title,
                    'chunk_index': len(chunks)
                }
            })
            
            # Start new chunk with overlap
            if chunk_overlap > 0 and len(chunks) > 0:
                # Get last few words from previous chunk for overlap
                words = current_chunk.split()
                overlap_words = words[-chunk_overlap//10:] if len(words) > chunk_overlap//10 else words
                current_chunk = ' '.join(overlap_words) + ' ' + sentence
                current_length = len(current_chunk)
            else:
                current_chunk = sentence
                current_length = sentence_length
        else:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += ' ' + sentence
            else:
                current_chunk = sentence
            current_length = len(current_chunk)
    
    # Add the last chunk if it exists
    if current_chunk.strip():
        chunks.append({
            'text': current_chunk.strip(),
            'metadata': {
                'url': url,
                'title': title,
                'chunk_index': len(chunks)
            }
        })
    
    return chunks

