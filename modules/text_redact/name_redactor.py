"""
Name redaction using Named Entity Recognition (SpaCy).

This module implements name detection and redaction using SpaCy's NER
for identifying and redacting personal names in text.
"""

import re
from typing import List, Tuple, Optional, Dict, Any, Set
import logging

logger = logging.getLogger(__name__)

# Try to import SpaCy
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("SpaCy not available. Install spacy for NER functionality.")


class NameRedactor:
    """
    Name redactor using SpaCy's Named Entity Recognition.
    
    This class provides name detection and redaction functionality using SpaCy's
    NER model to identify and redact personal names in text.
    """
    
    def __init__(self, model_name: str = 'en_core_web_sm', custom_patterns: List[str] = None):
        """
        Initialize the name redactor.
        
        Args:
            model_name: SpaCy model name to use
            custom_patterns: List of custom regex patterns for name detection
        """
        self.model_name = model_name
        self.custom_patterns = custom_patterns or []
        self.nlp = None
        self.spacy_available = SPACY_AVAILABLE
        
        if self.spacy_available:
            self._load_spacy_model()
        
        # Compile custom patterns
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.custom_patterns]
        
        logger.info(f"NameRedactor initialized with model: {model_name}")
    
    def _load_spacy_model(self) -> None:
        """Load SpaCy NER model."""
        try:
            self.nlp = spacy.load(self.model_name)
            logger.info(f"SpaCy model {self.model_name} loaded successfully")
        except OSError:
            logger.error(f"SpaCy model {self.model_name} not found. Please install it with: python -m spacy download {self.model_name}")
            self.spacy_available = False
        except Exception as e:
            logger.error(f"Error loading SpaCy model: {e}")
            self.spacy_available = False
    
    def detect_names(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect personal names in text using NER and custom patterns.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of name detection dictionaries with 'text', 'start', 'end', 'label', and 'confidence' keys
        """
        detections = []
        
        # Use SpaCy NER if available
        if self.spacy_available and self.nlp is not None:
            spacy_detections = self._detect_names_spacy(text)
            detections.extend(spacy_detections)
        
        # Use custom patterns
        custom_detections = self._detect_names_custom(text)
        detections.extend(custom_detections)
        
        # Remove duplicates and overlapping detections
        detections = self._deduplicate_detections(detections)
        
        logger.info(f"Detected {len(detections)} names in text")
        return detections
    
    def _detect_names_spacy(self, text: str) -> List[Dict[str, Any]]:
        """Detect names using SpaCy NER."""
        try:
            doc = self.nlp(text)
            detections = []
            
            for ent in doc.ents:
                # Focus on person names
                if ent.label_ in ['PERSON', 'PER']:
                    detections.append({
                        'text': ent.text,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'label': ent.label_,
                        'confidence': 0.9,  # SpaCy doesn't provide confidence scores by default
                        'method': 'spacy'
                    })
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in SpaCy name detection: {e}")
            return []
    
    def _detect_names_custom(self, text: str) -> List[Dict[str, Any]]:
        """Detect names using custom regex patterns."""
        detections = []
        
        for i, pattern in enumerate(self.compiled_patterns):
            for match in pattern.finditer(text):
                detections.append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'label': 'CUSTOM',
                    'confidence': 0.7,  # Lower confidence for custom patterns
                    'method': 'custom'
                })
        
        return detections
    
    def _deduplicate_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate and overlapping detections.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Deduplicated list of detections
        """
        if not detections:
            return []
        
        # Sort by start position
        detections.sort(key=lambda x: x['start'])
        
        deduplicated = []
        for detection in detections:
            # Check if this detection overlaps with any existing detection
            overlaps = False
            for existing in deduplicated:
                if (detection['start'] < existing['end'] and detection['end'] > existing['start']):
                    overlaps = True
                    # Keep the detection with higher confidence
                    if detection['confidence'] > existing['confidence']:
                        deduplicated.remove(existing)
                        deduplicated.append(detection)
                    break
            
            if not overlaps:
                deduplicated.append(detection)
        
        return deduplicated
    
    def redact_names(self, text: str, redaction_char: str = '*', 
                    redaction_length: int = None) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Redact personal names in text.
        
        Args:
            text: Input text to redact
            redaction_char: Character to use for redaction
            redaction_length: Length of redaction. If None, uses original text length
            
        Returns:
            Tuple of (redacted_text, list_of_redacted_names)
        """
        detections = self.detect_names(text)
        
        if not detections:
            return text, []
        
        # Sort detections by start position in reverse order to avoid index shifting
        detections.sort(key=lambda x: x['start'], reverse=True)
        
        redacted_text = text
        redacted_names = []
        
        for detection in detections:
            original_text = detection['text']
            start = detection['start']
            end = detection['end']
            
            # Determine redaction length
            if redaction_length is None:
                redaction_length = len(original_text)
            
            # Create redaction string
            redaction = redaction_char * redaction_length
            
            # Replace the name with redaction
            redacted_text = redacted_text[:start] + redaction + redacted_text[end:]
            
            # Track redacted names
            redacted_names.append({
                'original': original_text,
                'redacted': redaction,
                'start': start,
                'end': end,
                'label': detection['label'],
                'confidence': detection['confidence']
            })
        
        logger.info(f"Redacted {len(redacted_names)} names in text")
        return redacted_text, redacted_names
    
    def get_name_statistics(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about name detections.
        
        Args:
            detections: List of name detection dictionaries
            
        Returns:
            Dictionary containing name detection statistics
        """
        if not detections:
            return {
                'total_names': 0,
                'unique_names': 0,
                'average_confidence': 0.0,
                'label_distribution': {}
            }
        
        # Count unique names
        unique_names = set(det['text'].lower() for det in detections)
        
        # Count by label
        label_counts = {}
        for det in detections:
            label = det['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Calculate confidence statistics
        confidences = [det['confidence'] for det in detections]
        
        return {
            'total_names': len(detections),
            'unique_names': len(unique_names),
            'average_confidence': sum(confidences) / len(confidences),
            'max_confidence': max(confidences),
            'min_confidence': min(confidences),
            'label_distribution': label_counts
        }
    
    def add_custom_pattern(self, pattern: str) -> None:
        """
        Add a custom regex pattern for name detection.
        
        Args:
            pattern: Regex pattern to add
        """
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            self.compiled_patterns.append(compiled_pattern)
            self.custom_patterns.append(pattern)
            logger.info(f"Added custom pattern: {pattern}")
        except re.error as e:
            logger.error(f"Invalid regex pattern: {pattern}, error: {e}")
    
    def is_name(self, text: str) -> bool:
        """
        Check if a given text is likely a personal name.
        
        Args:
            text: Text to check
            
        Returns:
            True if text is likely a name, False otherwise
        """
        detections = self.detect_names(text)
        return len(detections) > 0
