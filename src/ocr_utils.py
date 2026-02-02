import re
import time
from typing import Optional, Tuple

# Updated for Indian plates (standard + high security + commercial)
PLATE_PATTERNS = [
    # Standard: MH12AB1234
    r"[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}",
    # With spaces: MH 12 AB 1234
    r"[A-Z]{2}\s+[0-9]{2}\s+[A-Z]{1,2}\s+[0-9]{4}",
    # High security: 1234 AB 12 MH
    r"[0-9]{4}\s+[A-Z]{2}\s+[0-9]{2}\s+[A-Z]{2}",
]

PLATE_REGEXES = [re.compile(pattern) for pattern in PLATE_PATTERNS]

def clean_plate(text: str) -> Optional[str]:
    """Clean and validate plate with confidence scoring."""
    if not text or len(text.strip()) < 6:
        return None
    
    # Clean: remove non-alphanumeric, normalize case and spaces
    cleaned = re.sub(r'[^A-Z0-9\s]', '', text.upper())
    cleaned = re.sub(r'\s+', ' ', cleaned.strip())
    
    # Try each pattern
    best_match = None
    best_score = 0
    
    for regex in PLATE_REGEXES:
        match = regex.search(cleaned)
        if match:
            candidate = match.group().replace(' ', '')  # Normalized form
            score = len(candidate) / 10.0  # Favor longer matches (8-10 chars)
            if score > best_score:
                best_score = score
                best_match = candidate
    
    # Accept matches of 8+ chars with reasonable confidence
    return best_match if best_match and len(best_match) >= 8 else None

def validate_plate_confidence(plate: str) -> Tuple[bool, float]:
    """Additional confidence check for logging."""
    if len(plate) < 8:
        return False, 0.0
    
    # Character distribution check (shouldn't be all numbers/letters)
    letters = sum(1 for c in plate if c.isalpha())
    numbers = sum(1 for c in plate if c.isdigit())
    
    if letters < 3 or numbers < 4:
        return False, 0.3
    
    # State codes check (first 2 chars)
    state_codes = {'MH', 'DL', 'KA', 'TN', 'GJ', 'PB', 'HR', 'UP', 'RJ', 'MP', 'UK', 'HP', 'JK', 'CH', 'PN', 'AP', 'TS', 'KL', 'BR', 'OR', 'WB', 'AS', 'CG', 'AR', 'SK', 'MZ', 'NL', 'MN', 'TR', 'NG', 'LD'}
    state = plate[:2]
    
    confidence = 0.8 if state in state_codes else 0.6
    confidence *= min(letters/3, numbers/4, 1.0)  # Balance check
    
    return True, confidence

class PlateTracker:
    def __init__(self, cooldown: float = 60.0):
        self.seen = {}  # plate -> last_seen_timestamp
        self.cooldown = cooldown
        self.log_count = 0  # Track total logs for debugging

    def should_log(self, plate: str) -> bool:
        """Check if plate should be logged with cooldown protection."""
        if not plate or len(plate) < 8:
            return False
            
        now = time.time()
        
        # Cooldown check
        if plate in self.seen and now - self.seen[plate] < self.cooldown:
            return False
            
        # Update tracking
        self.seen[plate] = now
        self.log_count += 1
        
        # Memory management: keep only recent plates
        if len(self.seen) > 1000:
            cutoff = now - 3600  # 1 hour
            self.seen = {k: v for k, v in self.seen.items() if v > cutoff}
        
        return True
    
    def get_stats(self) -> dict:
        """Get tracking statistics."""
        return {
            'total_logs': self.log_count,
            'active_plates': len(self.seen),
            'cooldown_seconds': self.cooldown
        }
