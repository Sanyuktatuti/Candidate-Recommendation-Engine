"""
AI configuration management with environment variable support.
Provides configurable parameters for temperature, max tokens, models, etc.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class AIModelConfig:
    """Configuration for a specific AI model."""
    model_name: str
    temperature: float
    max_tokens: int
    top_p: float
    frequency_penalty: float
    presence_penalty: float


@dataclass
class PromptConfig:
    """Configuration for prompt behavior."""
    summary_length: int  # Number of sentences
    summary_type: str    # 'concise', 'detailed', 'executive'
    include_risk_assessment: bool
    pii_protection_level: str  # 'strict', 'moderate', 'basic'
    

class AIConfig:
    """
    Centralized AI configuration management.
    Loads settings from environment variables with sensible defaults.
    """
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        self.openai_config = self._load_openai_config()
        self.cohere_config = self._load_cohere_config()
        self.hugging_face_config = self._load_hugging_face_config()
        self.prompt_config = self._load_prompt_config()
        self.general_config = self._load_general_config()
    
    def _load_openai_config(self) -> AIModelConfig:
        """Load OpenAI-specific configuration."""
        return AIModelConfig(
            model_name=os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
            temperature=float(os.getenv('OPENAI_TEMPERATURE', '0.7')),
            max_tokens=int(os.getenv('OPENAI_MAX_TOKENS', '300')),
            top_p=float(os.getenv('OPENAI_TOP_P', '1.0')),
            frequency_penalty=float(os.getenv('OPENAI_FREQUENCY_PENALTY', '0.0')),
            presence_penalty=float(os.getenv('OPENAI_PRESENCE_PENALTY', '0.0'))
        )
    
    def _load_cohere_config(self) -> AIModelConfig:
        """Load Cohere-specific configuration."""
        return AIModelConfig(
            model_name=os.getenv('COHERE_MODEL', 'command-r'),
            temperature=float(os.getenv('COHERE_TEMPERATURE', '0.6')),
            max_tokens=int(os.getenv('COHERE_MAX_TOKENS', '250')),
            top_p=float(os.getenv('COHERE_TOP_P', '0.9')),
            frequency_penalty=float(os.getenv('COHERE_FREQUENCY_PENALTY', '0.0')),
            presence_penalty=float(os.getenv('COHERE_PRESENCE_PENALTY', '0.0'))
        )
    
    def _load_hugging_face_config(self) -> AIModelConfig:
        """Load Hugging Face-specific configuration."""
        return AIModelConfig(
            model_name=os.getenv('HF_MODEL', 'microsoft/DialoGPT-medium'),
            temperature=float(os.getenv('HF_TEMPERATURE', '0.8')),
            max_tokens=int(os.getenv('HF_MAX_TOKENS', '200')),
            top_p=float(os.getenv('HF_TOP_P', '0.95')),
            frequency_penalty=float(os.getenv('HF_FREQUENCY_PENALTY', '0.0')),
            presence_penalty=float(os.getenv('HF_PRESENCE_PENALTY', '0.0'))
        )
    
    def _load_prompt_config(self) -> PromptConfig:
        """Load prompt-specific configuration."""
        return PromptConfig(
            summary_length=int(os.getenv('PROMPT_SUMMARY_LENGTH', '3')),
            summary_type=os.getenv('PROMPT_SUMMARY_TYPE', 'professional'),
            include_risk_assessment=os.getenv('PROMPT_INCLUDE_RISK', 'true').lower() == 'true',
            pii_protection_level=os.getenv('PII_PROTECTION_LEVEL', 'strict')
        )
    
    def _load_general_config(self) -> Dict[str, Any]:
        """Load general AI configuration."""
        return {
            'enable_pii_protection': os.getenv('ENABLE_PII_PROTECTION', 'true').lower() == 'true',
            'max_input_length': int(os.getenv('MAX_INPUT_LENGTH', '5000')),
            'max_resume_length': int(os.getenv('MAX_RESUME_LENGTH', '3000')),
            'max_job_desc_length': int(os.getenv('MAX_JOB_DESC_LENGTH', '2000')),
            'fallback_on_error': os.getenv('FALLBACK_ON_ERROR', 'true').lower() == 'true',
            'log_pii_detections': os.getenv('LOG_PII_DETECTIONS', 'false').lower() == 'true',
            'analysis_timeout': int(os.getenv('ANALYSIS_TIMEOUT', '30')),  # seconds
            'retry_attempts': int(os.getenv('RETRY_ATTEMPTS', '3')),
            'retry_delay': float(os.getenv('RETRY_DELAY', '1.0'))  # seconds
        }
    
    def get_model_config(self, service: str) -> Optional[AIModelConfig]:
        """
        Get model configuration for a specific service.
        
        Args:
            service: Service name ('openai', 'cohere', 'hugging_face')
            
        Returns:
            AIModelConfig for the service or None if not found
        """
        service_map = {
            'openai': self.openai_config,
            'cohere': self.cohere_config,
            'hugging_face': self.hugging_face_config,
            'hf': self.hugging_face_config  # alias
        }
        
        return service_map.get(service.lower())
    
    def get_openai_params(self) -> Dict[str, Any]:
        """Get OpenAI API parameters as dictionary."""
        config = self.openai_config
        return {
            'model': config.model_name,
            'temperature': config.temperature,
            'max_tokens': config.max_tokens,
            'top_p': config.top_p,
            'frequency_penalty': config.frequency_penalty,
            'presence_penalty': config.presence_penalty
        }
    
    def get_cohere_params(self) -> Dict[str, Any]:
        """Get Cohere API parameters as dictionary."""
        config = self.cohere_config
        return {
            'model': config.model_name,
            'temperature': config.temperature,
            'max_tokens': config.max_tokens,
            'p': config.top_p,
            'frequency_penalty': config.frequency_penalty,
            'presence_penalty': config.presence_penalty
        }
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate configuration values and return validation results.
        
        Returns:
            Dictionary with validation results and any issues found
        """
        issues = []
        warnings = []
        
        # Validate temperature ranges
        for service, config in [
            ('openai', self.openai_config),
            ('cohere', self.cohere_config),
            ('hugging_face', self.hugging_face_config)
        ]:
            if not 0.0 <= config.temperature <= 2.0:
                issues.append(f"{service} temperature ({config.temperature}) should be between 0.0 and 2.0")
            
            if config.max_tokens <= 0:
                issues.append(f"{service} max_tokens ({config.max_tokens}) must be positive")
            
            if not 0.0 <= config.top_p <= 1.0:
                warnings.append(f"{service} top_p ({config.top_p}) typically ranges from 0.0 to 1.0")
        
        # Validate prompt config
        if self.prompt_config.summary_length <= 0:
            issues.append(f"summary_length ({self.prompt_config.summary_length}) must be positive")
        
        # Validate general config
        general = self.general_config
        if general['max_input_length'] <= 0:
            issues.append("max_input_length must be positive")
        
        if general['analysis_timeout'] <= 0:
            issues.append("analysis_timeout must be positive")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'config_summary': {
                'openai_model': self.openai_config.model_name,
                'summary_length': self.prompt_config.summary_length,
                'pii_protection': general['enable_pii_protection'],
                'fallback_enabled': general['fallback_on_error']
            }
        }
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration for logging/debugging."""
        return {
            'models': {
                'openai': f"{self.openai_config.model_name} (temp: {self.openai_config.temperature})",
                'cohere': f"{self.cohere_config.model_name} (temp: {self.cohere_config.temperature})",
                'hugging_face': f"{self.hugging_face_config.model_name} (temp: {self.hugging_face_config.temperature})"
            },
            'prompt_settings': {
                'summary_length': self.prompt_config.summary_length,
                'summary_type': self.prompt_config.summary_type,
                'pii_protection': self.prompt_config.pii_protection_level
            },
            'limits': {
                'max_input_length': self.general_config['max_input_length'],
                'analysis_timeout': self.general_config['analysis_timeout'],
                'retry_attempts': self.general_config['retry_attempts']
            }
        }


# Global configuration instance
_config_instance = None

def get_ai_config() -> AIConfig:
    """Get global AI configuration instance (singleton pattern)."""
    global _config_instance
    if _config_instance is None:
        _config_instance = AIConfig()
    return _config_instance

def reload_config() -> AIConfig:
    """Reload configuration from environment variables."""
    global _config_instance
    _config_instance = AIConfig()
    return _config_instance
