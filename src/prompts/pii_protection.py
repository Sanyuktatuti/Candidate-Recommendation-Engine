"""
PII (Personally Identifiable Information) protection system.
Detects and removes sensitive information before API calls.
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class PIIDetection:
    """Results of PII detection."""
    original_text: str
    cleaned_text: str
    detected_pii: List[Dict[str, str]]
    risk_level: str  # 'low', 'medium', 'high'


class PIIProtector:
    """
    Comprehensive PII detection and protection system.
    Removes sensitive information while preserving context for AI analysis.
    """
    
    def __init__(self):
        """Initialize PII protection patterns."""
        self.patterns = {
            'email': {
                'pattern': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'replacement': '[EMAIL_REDACTED]',
                'risk': 'high'
            },
            'phone': {
                'pattern': r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
                'replacement': '[PHONE_REDACTED]',
                'risk': 'high'
            },
            'ssn': {
                'pattern': r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b',
                'replacement': '[SSN_REDACTED]',
                'risk': 'high'
            },
            'address_street': {
                'pattern': r'\b\d{1,6}\s+[A-Za-z0-9\s,\.#-]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Circle|Cir|Way|Place|Pl)\b',
                'replacement': '[ADDRESS_REDACTED]',
                'risk': 'medium'
            },
            'zip_code': {
                'pattern': r'\b\d{5}(?:-\d{4})?\b',
                'replacement': '[ZIP_REDACTED]',
                'risk': 'low'
            },
            'credit_card': {
                'pattern': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
                'replacement': '[CARD_REDACTED]',
                'risk': 'high'
            },
            'date_of_birth': {
                'pattern': r'\b(?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b',
                'replacement': '[DOB_REDACTED]',
                'risk': 'high'
            },
            'license_number': {
                'pattern': r'\b[A-Z]{1,2}\d{6,8}\b',
                'replacement': '[LICENSE_REDACTED]',
                'risk': 'medium'
            }
        }
        
        # Sensitive keywords that might indicate PII context
        self.sensitive_keywords = [
            'social security', 'ssn', 'date of birth', 'dob', 'drivers license',
            'passport', 'home address', 'personal address', 'emergency contact',
            'next of kin', 'bank account', 'routing number', 'credit score'
        ]
    
    def detect_and_clean(self, text: str, preserve_structure: bool = True) -> PIIDetection:
        """
        Detect and remove PII from text.
        
        Args:
            text: Input text to clean
            preserve_structure: Whether to maintain text structure for analysis
            
        Returns:
            PIIDetection object with cleaned text and detection results
        """
        if not text or not isinstance(text, str):
            return PIIDetection(
                original_text=text or "",
                cleaned_text=text or "",
                detected_pii=[],
                risk_level='low'
            )
        
        cleaned_text = text
        detected_pii = []
        max_risk_level = 'low'
        
        # Apply PII detection patterns
        for pii_type, config in self.patterns.items():
            pattern = config['pattern']
            replacement = config['replacement']
            risk = config['risk']
            
            matches = re.finditer(pattern, cleaned_text, re.IGNORECASE)
            for match in matches:
                detected_pii.append({
                    'type': pii_type,
                    'text': match.group(),
                    'position': (match.start(), match.end()),
                    'risk': risk
                })
                
                # Update maximum risk level
                if risk == 'high' or (risk == 'medium' and max_risk_level == 'low'):
                    max_risk_level = risk
            
            # Replace matches
            cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE)
        
        # Check for sensitive keyword contexts
        for keyword in self.sensitive_keywords:
            if keyword.lower() in text.lower():
                detected_pii.append({
                    'type': 'sensitive_context',
                    'text': keyword,
                    'position': (-1, -1),
                    'risk': 'medium'
                })
                max_risk_level = 'medium' if max_risk_level == 'low' else max_risk_level
        
        # Additional cleaning for resume context
        if preserve_structure:
            cleaned_text = self._preserve_professional_context(cleaned_text)
        
        return PIIDetection(
            original_text=text,
            cleaned_text=cleaned_text,
            detected_pii=detected_pii,
            risk_level=max_risk_level
        )
    
    def _preserve_professional_context(self, text: str) -> str:
        """
        Preserve professional context while removing PII.
        Keeps skills, experience, education relevant for job matching.
        """
        # Remove lines that likely contain personal info
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Skip lines that likely contain personal information
            if any(keyword in line_lower for keyword in [
                'address:', 'phone:', 'email:', 'cell:', 'mobile:',
                'emergency contact', 'references available', 'personal information'
            ]):
                continue
            
            # Keep professional content
            if line.strip():  # Non-empty lines
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def get_risk_assessment(self, detection: PIIDetection) -> Dict[str, any]:
        """
        Provide risk assessment and recommendations.
        
        Args:
            detection: PIIDetection results
            
        Returns:
            Risk assessment with recommendations
        """
        high_risk_count = len([pii for pii in detection.detected_pii if pii['risk'] == 'high'])
        medium_risk_count = len([pii for pii in detection.detected_pii if pii['risk'] == 'medium'])
        
        recommendations = []
        
        if high_risk_count > 0:
            recommendations.append("High-risk PII detected. Review data handling policies.")
        
        if medium_risk_count > 2:
            recommendations.append("Multiple medium-risk elements found. Consider additional screening.")
        
        if detection.risk_level == 'high':
            recommendations.append("Recommend manual review before processing.")
        
        return {
            'risk_level': detection.risk_level,
            'high_risk_items': high_risk_count,
            'medium_risk_items': medium_risk_count,
            'total_detections': len(detection.detected_pii),
            'recommendations': recommendations,
            'safe_for_ai_processing': detection.risk_level != 'high' or high_risk_count == 0
        }
    
    def validate_clean_text(self, text: str) -> bool:
        """
        Validate that text is clean and safe for AI processing.
        
        Args:
            text: Text to validate
            
        Returns:
            True if text appears PII-free
        """
        detection = self.detect_and_clean(text, preserve_structure=False)
        risk_assessment = self.get_risk_assessment(detection)
        
        return risk_assessment['safe_for_ai_processing']


# Convenience functions for easy import
def clean_text_for_ai(text: str) -> str:
    """Quick function to clean text for AI processing."""
    protector = PIIProtector()
    detection = protector.detect_and_clean(text)
    return detection.cleaned_text


def assess_pii_risk(text: str) -> Dict[str, any]:
    """Quick function to assess PII risk in text."""
    protector = PIIProtector()
    detection = protector.detect_and_clean(text)
    return protector.get_risk_assessment(detection)
