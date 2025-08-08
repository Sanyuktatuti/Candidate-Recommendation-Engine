"""
Comprehensive prompt management system.
Handles template loading, PII protection, and parameterized prompt generation.
"""

import os
import re
from typing import Dict, Any, Optional, List
from pathlib import Path

from .pii_protection import PIIProtector, PIIDetection
from .config import get_ai_config, AIConfig


class PromptTemplate:
    """Represents a prompt template with metadata and validation."""
    
    def __init__(self, name: str, content: str, required_params: List[str] = None):
        """
        Initialize prompt template.
        
        Args:
            name: Template name/identifier
            content: Template content with {parameter} placeholders
            required_params: List of required parameter names
        """
        self.name = name
        self.content = content
        self.required_params = required_params or []
        self._discover_parameters()
    
    def _discover_parameters(self) -> None:
        """Automatically discover parameters in template content."""
        # Find all {parameter} patterns
        pattern = r'\{([^}]+)\}'
        discovered_params = set(re.findall(pattern, self.content))
        
        # Update required params if not explicitly set
        if not self.required_params:
            self.required_params = list(discovered_params)
        
        self.all_parameters = discovered_params
    
    def validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that all required parameters are provided.
        
        Args:
            params: Dictionary of parameters to validate
            
        Returns:
            Validation results with missing parameters and issues
        """
        provided_params = set(params.keys())
        required_params = set(self.required_params)
        missing_params = required_params - provided_params
        extra_params = provided_params - self.all_parameters
        
        return {
            'valid': len(missing_params) == 0,
            'missing_params': list(missing_params),
            'extra_params': list(extra_params),
            'all_provided': len(missing_params) == 0 and len(extra_params) == 0
        }
    
    def render(self, params: Dict[str, Any]) -> str:
        """
        Render template with provided parameters.
        
        Args:
            params: Dictionary of parameter values
            
        Returns:
            Rendered template content
            
        Raises:
            ValueError: If required parameters are missing
        """
        validation = self.validate_parameters(params)
        
        if not validation['valid']:
            missing = ', '.join(validation['missing_params'])
            raise ValueError(f"Missing required parameters for template '{self.name}': {missing}")
        
        try:
            return self.content.format(**params)
        except KeyError as e:
            raise ValueError(f"Template parameter error in '{self.name}': {e}")


class PromptManager:
    """
    Centralized prompt management with PII protection and configuration.
    Handles template loading, parameter validation, and safe prompt generation.
    """
    
    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize prompt manager.
        
        Args:
            templates_dir: Directory containing prompt templates
        """
        self.config = get_ai_config()
        self.pii_protector = PIIProtector()
        
        # Set templates directory
        if templates_dir:
            self.templates_dir = Path(templates_dir)
        else:
            # Default to templates directory relative to this file
            current_dir = Path(__file__).parent
            self.templates_dir = current_dir / 'templates'
        
        # Load templates
        self.templates: Dict[str, PromptTemplate] = {}
        self._load_templates()
    
    def _load_templates(self) -> None:
        """Load all prompt templates from the templates directory."""
        if not self.templates_dir.exists():
            raise FileNotFoundError(f"Templates directory not found: {self.templates_dir}")
        
        template_files = self.templates_dir.glob('*.txt')
        
        for template_file in template_files:
            try:
                template_name = template_file.stem
                content = template_file.read_text(encoding='utf-8')
                
                # Create template object
                template = PromptTemplate(template_name, content)
                self.templates[template_name] = template
                
            except Exception as e:
                # Log error but continue loading other templates
                print(f"Warning: Failed to load template {template_file}: {e}")
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """
        Get a specific template by name.
        
        Args:
            name: Template name
            
        Returns:
            PromptTemplate object or None if not found
        """
        return self.templates.get(name)
    
    def list_templates(self) -> List[str]:
        """Get list of available template names."""
        return list(self.templates.keys())
    
    def generate_candidate_analysis(
        self, 
        job_description: str, 
        resume_text: str, 
        candidate_name: str = "Candidate",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate candidate analysis prompt with PII protection.
        
        Args:
            job_description: Job description text
            resume_text: Candidate resume text
            candidate_name: Candidate name (optional)
            **kwargs: Additional template parameters
            
        Returns:
            Dictionary with prompt, PII detection results, and metadata
        """
        return self._generate_protected_prompt(
            template_name='candidate_analysis',
            job_description=job_description,
            resume_text=resume_text,
            candidate_name=candidate_name,
            **kwargs
        )
    
    def generate_job_matching(
        self, 
        job_description: str, 
        resume_text: str,
        skills_weight: int = 30,
        experience_weight: int = 35,
        qualification_weight: int = 20,
        impact_weight: int = 15,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate job matching analysis prompt.
        
        Args:
            job_description: Job description text
            resume_text: Candidate resume text
            skills_weight: Weight for skills assessment (%)
            experience_weight: Weight for experience assessment (%)
            qualification_weight: Weight for qualification assessment (%)
            impact_weight: Weight for impact assessment (%)
            **kwargs: Additional template parameters
            
        Returns:
            Dictionary with prompt, PII detection results, and metadata
        """
        return self._generate_protected_prompt(
            template_name='job_matching',
            job_description=job_description,
            resume_text=resume_text,
            skills_weight=skills_weight,
            experience_weight=experience_weight,
            qualification_weight=qualification_weight,
            impact_weight=impact_weight,
            **kwargs
        )
    
    def generate_summary(
        self, 
        job_description: str, 
        resume_text: str,
        summary_type: str = None,
        summary_length: int = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate candidate summary prompt.
        
        Args:
            job_description: Job description text
            resume_text: Candidate resume text
            summary_type: Type of summary ('concise', 'detailed', 'executive')
            summary_length: Number of sentences for summary
            **kwargs: Additional template parameters
            
        Returns:
            Dictionary with prompt, PII detection results, and metadata
        """
        # Use config defaults if not provided
        if summary_type is None:
            summary_type = self.config.prompt_config.summary_type
        if summary_length is None:
            summary_length = self.config.prompt_config.summary_length
        
        return self._generate_protected_prompt(
            template_name='summary_generation',
            job_description=job_description,
            resume_text=resume_text,
            summary_type=summary_type,
            summary_length=summary_length,
            **kwargs
        )
    
    def _generate_protected_prompt(
        self, 
        template_name: str, 
        job_description: str, 
        resume_text: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate prompt with PII protection and configuration.
        
        Args:
            template_name: Name of template to use
            job_description: Job description text
            resume_text: Resume text
            **kwargs: Additional template parameters
            
        Returns:
            Dictionary with prompt, PII detection, and metadata
        """
        # Get template
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        # Apply PII protection if enabled
        if self.config.general_config['enable_pii_protection']:
            job_detection = self.pii_protector.detect_and_clean(job_description)
            resume_detection = self.pii_protector.detect_and_clean(resume_text)
            
            clean_job_desc = job_detection.cleaned_text
            clean_resume = resume_detection.cleaned_text
        else:
            job_detection = PIIDetection(job_description, job_description, [], 'low')
            resume_detection = PIIDetection(resume_text, resume_text, [], 'low')
            
            clean_job_desc = job_description
            clean_resume = resume_text
        
        # Truncate texts based on configuration
        max_job_length = self.config.general_config['max_job_desc_length']
        max_resume_length = self.config.general_config['max_resume_length']
        
        if len(clean_job_desc) > max_job_length:
            clean_job_desc = clean_job_desc[:max_job_length] + "..."
        
        if len(clean_resume) > max_resume_length:
            clean_resume = clean_resume[:max_resume_length] + "..."
        
        # Prepare template parameters
        template_params = {
            'job_description': clean_job_desc,
            'resume_text': clean_resume,
            **kwargs
        }
        
        # Add configuration-based parameters
        template_params.update({
            'summary_length': self.config.prompt_config.summary_length,
            'summary_type': self.config.prompt_config.summary_type
        })
        
        # Render template
        try:
            rendered_prompt = template.render(template_params)
        except ValueError as e:
            return {
                'success': False,
                'error': str(e),
                'template_name': template_name
            }
        
        # Prepare response
        return {
            'success': True,
            'prompt': rendered_prompt,
            'template_name': template_name,
            'pii_detection': {
                'job_description': {
                    'risk_level': job_detection.risk_level,
                    'detected_count': len(job_detection.detected_pii),
                    'original_length': len(job_description),
                    'cleaned_length': len(clean_job_desc)
                },
                'resume_text': {
                    'risk_level': resume_detection.risk_level,
                    'detected_count': len(resume_detection.detected_pii),
                    'original_length': len(resume_text),
                    'cleaned_length': len(clean_resume)
                }
            },
            'metadata': {
                'config_used': self.config.prompt_config.summary_type,
                'pii_protection_enabled': self.config.general_config['enable_pii_protection'],
                'template_parameters': list(template_params.keys()),
                'truncated': {
                    'job_description': len(job_description) > max_job_length,
                    'resume_text': len(resume_text) > max_resume_length
                }
            }
        }
    
    def validate_template_parameters(self, template_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parameters for a specific template.
        
        Args:
            template_name: Name of template
            params: Parameters to validate
            
        Returns:
            Validation results
        """
        template = self.get_template(template_name)
        if not template:
            return {
                'valid': False,
                'error': f"Template '{template_name}' not found"
            }
        
        return template.validate_parameters(params)
    
    def get_template_info(self, template_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific template.
        
        Args:
            template_name: Name of template
            
        Returns:
            Template information or None if not found
        """
        template = self.get_template(template_name)
        if not template:
            return None
        
        return {
            'name': template.name,
            'required_parameters': template.required_params,
            'all_parameters': list(template.all_parameters),
            'content_length': len(template.content),
            'content_preview': template.content[:200] + "..." if len(template.content) > 200 else template.content
        }
    
    def reload_templates(self) -> Dict[str, Any]:
        """
        Reload all templates from disk.
        
        Returns:
            Reload results with success/failure information
        """
        old_templates = list(self.templates.keys())
        self.templates.clear()
        
        try:
            self._load_templates()
            new_templates = list(self.templates.keys())
            
            return {
                'success': True,
                'old_templates': old_templates,
                'new_templates': new_templates,
                'added': [t for t in new_templates if t not in old_templates],
                'removed': [t for t in old_templates if t not in new_templates],
                'total_loaded': len(new_templates)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'templates_loaded': len(self.templates)
            }


# Global prompt manager instance
_prompt_manager_instance = None

def get_prompt_manager() -> PromptManager:
    """Get global prompt manager instance (singleton pattern)."""
    global _prompt_manager_instance
    if _prompt_manager_instance is None:
        _prompt_manager_instance = PromptManager()
    return _prompt_manager_instance

def reload_prompt_manager() -> PromptManager:
    """Reload prompt manager and templates."""
    global _prompt_manager_instance
    _prompt_manager_instance = PromptManager()
    return _prompt_manager_instance
