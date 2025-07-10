import os
import json
import hashlib
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Tuple

# === CONFIGURATION ===
SOURCE_DIR = "data"
OUTPUT_FILE = "chunks.json"
CHUNK_SIZE = 800  # Increased for better context
CHUNK_OVERLAP = 200  # More overlap for continuity
MIN_CHUNK_SIZE = 100  # Don't create tiny chunks

# === Semantic section markers ===
SECTION_MARKERS = [
    # Headers
    r'^#{1,6}\s+.*$',  # Markdown headers
    r'^[A-Z][A-Z\s]{2,}:?\s*$',  # ALL CAPS headers
    r'^\d+\.\s+[A-Z].*$',  # Numbered sections
    # Content breaks
    r'^-{3,}$',  # Horizontal rules
    r'^\*{3,}$',  # Asterisk breaks
    # Semantic indicators
    r'^(Introduction|Overview|Summary|Conclusion|Services|Products|About|Contact|FAQ)',
    r'^(Our|We|The company|Our team|Our services|Our products)',
]

def is_section_boundary(line: str) -> bool:
    """Check if a line represents a section boundary"""
    line = line.strip()
    for pattern in SECTION_MARKERS:
        if re.match(pattern, line, re.IGNORECASE):
            return True
    return False

def smart_split_text(text: str) -> List[str]:
    """Split text at semantic boundaries when possible"""
    lines = text.split('\n')
    sections = []
    current_section = []
    
    for i, line in enumerate(lines):
        if is_section_boundary(line) and current_section:
            # Save current section
            sections.append('\n'.join(current_section))
            current_section = [line]
        else:
            current_section.append(line)
    
    # Don't forget the last section
    if current_section:
        sections.append('\n'.join(current_section))
    
    return sections

def enhance_chunk_with_context(chunk: str, prev_chunk: str = "", next_chunk: str = "") -> str:
    """Add context indicators to chunks for better retrieval"""
    enhanced = chunk
    
    # Add continuation markers if needed
    if prev_chunk and not chunk[0].isupper():
        enhanced = "[...continued] " + enhanced
    
    # Check if chunk ends mid-sentence
    if next_chunk and chunk and chunk[-1] not in '.!?":':
        enhanced = enhanced + " [continues...]"
    
    return enhanced

def create_metadata_rich_chunks(filename: str, content: str) -> List[Dict]:
    """Create chunks with rich metadata"""
    # First, try semantic splitting
    semantic_sections = smart_split_text(content)
    
    # Then use RecursiveCharacterTextSplitter on each section
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    
    all_chunks = []
    chunk_id = 0
    
    for section_idx, section in enumerate(semantic_sections):
        if len(section.strip()) < MIN_CHUNK_SIZE:
            continue
            
        # Split large sections
        if len(section) > CHUNK_SIZE:
            sub_chunks = splitter.split_text(section)
        else:
            sub_chunks = [section]
        
        for sub_idx, chunk_text in enumerate(sub_chunks):
            if len(chunk_text.strip()) < MIN_CHUNK_SIZE:
                continue
            
            # Detect chunk type/topic
            chunk_lower = chunk_text.lower()
            topics = []
            
            # Service-related keywords
            if any(kw in chunk_lower for kw in ['service', 'solution', 'offer', 'provide', 'help']):
                topics.append('services')
            if any(kw in chunk_lower for kw in ['seo', 'search engine', 'optimization', 'ranking']):
                topics.append('seo')
            if any(kw in chunk_lower for kw in ['website', 'web design', 'development', 'responsive']):
                topics.append('web_development')
            if any(kw in chunk_lower for kw in ['advertising', 'ads', 'ppc', 'campaign', 'meta', 'google ads']):
                topics.append('advertising')
            if any(kw in chunk_lower for kw in ['social media', 'facebook', 'instagram', 'linkedin']):
                topics.append('social_media')
            
            # Business info keywords
            if any(kw in chunk_lower for kw in ['about', 'founded', 'mission', 'vision', 'team']):
                topics.append('about')
            if any(kw in chunk_lower for kw in ['contact', 'email', 'phone', 'address', 'reach']):
                topics.append('contact')
            if any(kw in chunk_lower for kw in ['price', 'cost', 'fee', 'pricing', 'package', '$']):
                topics.append('pricing')
            if any(kw in chunk_lower for kw in ['faq', 'question', 'answer', 'how to', 'what is']):
                topics.append('faq')
            
            # Industry-specific
            if any(kw in chunk_lower for kw in ['restaurant', 'food', 'menu', 'dining']):
                topics.append('restaurant')
            if any(kw in chunk_lower for kw in ['fitness', 'gym', 'workout', 'health']):
                topics.append('fitness')
            if any(kw in chunk_lower for kw in ['law', 'legal', 'attorney', 'lawyer']):
                topics.append('legal')
            if any(kw in chunk_lower for kw in ['real estate', 'property', 'listing', 'agent']):
                topics.append('real_estate')
            
            chunk_data = {
                "id": f"{filename}-s{section_idx}-c{chunk_id}",
                "text": chunk_text.strip(),
                "source": filename,
                "section_index": section_idx,
                "chunk_index": chunk_id,
                "topics": topics if topics else ["general"],
                "has_contact_info": bool(re.search(r'[\w\.-]+@[\w\.-]+|phone:|tel:|address:', chunk_lower)),
                "has_pricing": bool(re.search(r'\$\d+|price|cost|fee', chunk_lower)),
                "word_count": len(chunk_text.split()),
            }
            
            all_chunks.append(chunk_data)
            chunk_id += 1
    
    return all_chunks

def load_texts(source_folder: str) -> List[Tuple[str, str]]:
    """Load all text files from source folder and subdirectories"""
    files_data = []
    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:  # Only add non-empty files
                            # Include subdirectory in filename for context
                            relative_path = os.path.relpath(path, source_folder)
                            files_data.append((relative_path, content))
                except Exception as e:
                    print(f"[-] Error reading {path}: {e}")
    return files_data

def deduplicate_chunks(chunks: List[Dict]) -> List[Dict]:
    """Remove duplicate chunks while preserving metadata"""
    seen_hashes = {}
    unique_chunks = []
    
    for chunk in chunks:
        # Create hash of normalized text
        normalized_text = ' '.join(chunk['text'].lower().split())
        text_hash = hashlib.md5(normalized_text.encode('utf-8')).hexdigest()
        
        if text_hash not in seen_hashes:
            seen_hashes[text_hash] = chunk['id']
            unique_chunks.append(chunk)
        else:
            # Merge topics from duplicate
            original_id = seen_hashes[text_hash]
            for unique_chunk in unique_chunks:
                if unique_chunk['id'] == original_id:
                    # Merge topics
                    existing_topics = set(unique_chunk.get('topics', []))
                    new_topics = set(chunk.get('topics', []))
                    unique_chunk['topics'] = list(existing_topics.union(new_topics))
                    break
    
    return unique_chunks

def main():
    print("[+] Loading text files...")
    files_data = load_texts(SOURCE_DIR)
    print(f"[+] Found {len(files_data)} files")
    
    all_chunks = []
    for filename, content in files_data:
        print(f"[+] Processing {filename}...")
        chunks = create_metadata_rich_chunks(filename, content)
        all_chunks.extend(chunks)
    
    print(f"[+] Created {len(all_chunks)} total chunks")
    
    # Deduplicate
    unique_chunks = deduplicate_chunks(all_chunks)
    print(f"[+] After deduplication: {len(unique_chunks)} unique chunks")
    
    # Save with pretty formatting
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(unique_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"[✓] Chunks saved to {OUTPUT_FILE}")
    
    # Print statistics
    topic_counts = {}
    for chunk in unique_chunks:
        for topic in chunk.get('topics', ['general']):
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    print("\n[+] Topic distribution:")
    for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {topic}: {count} chunks")

if __name__ == "__main__":
    main()