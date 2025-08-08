"""
Unified AI service with automatic hierarchy: OpenAI → Enhanced Analysis.
Now includes comprehensive prompt hygiene with PII protection and configurable parameters.
"""

import os
import re
import time
from typing import List, Dict, Any, Optional
import streamlit as st

# Load environment variables for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, that's fine
    pass

# Import prompt management system
from ..prompts import PromptManager, PIIProtector
from ..prompts.config import get_ai_config

try:
    import openai
except ImportError:
    openai = None


class UnifiedAIService:
    """
    Unified AI service with automatic hierarchy: OpenAI → Enhanced Analysis.
    
    Features:
    - Comprehensive prompt hygiene with PII protection
    - Configurable AI parameters via environment variables
    - Template-based prompt management
    - Intelligent fallback system
    - Security-first design for HR/recruiting use cases
    """
    
    def __init__(self):
        """Initialize the AI service with prompt hygiene system."""
        # Initialize configuration and prompt management
        self.config = get_ai_config()
        self.prompt_manager = PromptManager()
        self.pii_protector = PIIProtector()
        
        # Load API key with priority system
        self._load_api_keys()
        
        # Initialize OpenAI client if available
        self.openai_client = None
        if self.openai_api_key and openai:
            try:
                self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
            except Exception as e:
                print(f"Warning: Failed to initialize OpenAI client: {e}")
                self.openai_client = None
    
    def _load_api_keys(self) -> None:
        """Load API keys with priority: environment variables → Streamlit secrets."""
        # Priority 1: Environment variables (for local development)
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        
        # Priority 2: Streamlit secrets (for cloud deployment) - only if env var is empty
        if not self.openai_api_key:
            try:
                self.openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
            except Exception:
                # No secrets available, use empty string (will fallback to enhanced analysis)
                self.openai_api_key = ""
        
        self.job_domains = {
            'data': ['data', 'analytics', 'scientist', 'analysis', 'statistics', 'research'],
            'engineering': ['software', 'engineer', 'developer', 'programming', 'coding', 'technical'],
            'management': ['manager', 'lead', 'director', 'supervisor', 'management', 'leadership'],
            'marketing': ['marketing', 'sales', 'business', 'customer', 'growth', 'strategy'],
            'design': ['design', 'ui', 'ux', 'creative', 'visual', 'graphics'],
            'operations': ['operations', 'process', 'supply', 'logistics', 'efficiency'],
            'finance': ['finance', 'accounting', 'financial', 'budget', 'investment', 'risk'],
            'hr': ['human resources', 'hr', 'recruitment', 'talent', 'people', 'culture']
        }
    
    def generate_fit_summary(
        self, 
        job_description: str, 
        resume_text: str, 
        candidate_name: str = "Candidate",
        summary_type: str = None
    ) -> Dict[str, Any]:
        """
        Generate AI-powered fit summary with comprehensive prompt hygiene.
        
        Args:
            job_description: Job description text
            resume_text: Candidate resume text
            candidate_name: Candidate name for personalization
            summary_type: Type of summary ('concise', 'detailed', 'professional')
            
        Returns:
            Dictionary with summary, metadata, and PII detection results
        """
        start_time = time.time()
        
        # Use OpenAI if available, otherwise fallback to enhanced analysis
        if self.openai_client:
            try:
                result = self._generate_openai_summary_with_hygiene(
                    job_description, resume_text, candidate_name, summary_type
                )
                result['service_used'] = 'openai'
                result['processing_time'] = time.time() - start_time
                return result
            except Exception as e:
                # Log error and fallback
                print(f"OpenAI analysis failed: {e}")
        
        # Fallback to enhanced analysis with prompt hygiene
        result = self._generate_enhanced_summary_with_hygiene(
            job_description, resume_text, candidate_name, summary_type
        )
        result['service_used'] = 'enhanced_analysis'
        result['processing_time'] = time.time() - start_time
        return result
    
    def generate_fit_summary_legacy(self, job_description: str, resume_text: str, candidate_name: str) -> str:
        """
        Legacy method for backward compatibility.
        Returns just the summary text instead of full metadata.
        """
        result = self.generate_fit_summary(job_description, resume_text, candidate_name)
        return result.get('summary', result.get('error', 'Unable to generate summary'))
    
    def _generate_openai_summary_with_hygiene(
        self, 
        job_description: str, 
        resume_text: str, 
        candidate_name: str,
        summary_type: str = None
    ) -> Dict[str, Any]:
        """Generate OpenAI summary with prompt hygiene and PII protection."""
        try:
            # Generate prompt using prompt manager with PII protection
            prompt_result = self.prompt_manager.generate_summary(
                job_description=job_description,
                resume_text=resume_text,
                summary_type=summary_type or 'professional'
            )
            
            if not prompt_result['success']:
                return {
                    'success': False,
                    'error': prompt_result.get('error', 'Failed to generate prompt'),
                    'service_used': 'openai'
                }
            
            # Get OpenAI configuration
            openai_params = self.config.get_openai_params()
            
            # Make OpenAI API call with configured parameters
            response = self.openai_client.chat.completions.create(
                model=openai_params['model'],
                messages=[
                    {"role": "system", "content": "You are an expert HR analyst specializing in candidate evaluation."},
                    {"role": "user", "content": prompt_result['prompt']}
                ],
                temperature=openai_params['temperature'],
                max_tokens=openai_params['max_tokens'],
                top_p=openai_params['top_p'],
                frequency_penalty=openai_params['frequency_penalty'],
                presence_penalty=openai_params['presence_penalty']
            )
            
            summary = response.choices[0].message.content.strip()
            
            return {
                'success': True,
                'summary': summary,
                'candidate_name': candidate_name,
                'pii_detection': prompt_result['pii_detection'],
                'metadata': {
                    **prompt_result['metadata'],
                    'openai_model': openai_params['model'],
                    'openai_params': openai_params,
                    'tokens_used': response.usage.total_tokens if hasattr(response, 'usage') else None
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"OpenAI API error: {str(e)}",
                'candidate_name': candidate_name,
                'fallback_attempted': True
            }
    
    def _generate_enhanced_summary_with_hygiene(
        self, 
        job_description: str, 
        resume_text: str, 
        candidate_name: str,
        summary_type: str = None
    ) -> Dict[str, Any]:
        """Generate enhanced analysis summary with prompt hygiene and PII protection."""
        try:
            # Apply PII protection if enabled
            if self.config.general_config['enable_pii_protection']:
                job_detection = self.pii_protector.detect_and_clean(job_description)
                resume_detection = self.pii_protector.detect_and_clean(resume_text)
                
                clean_job_desc = job_detection.cleaned_text
                clean_resume = resume_detection.cleaned_text
            else:
                # Create mock detection results
                from ..prompts.pii_protection import PIIDetection
                job_detection = PIIDetection(job_description, job_description, [], 'low')
                resume_detection = PIIDetection(resume_text, resume_text, [], 'low')
                
                clean_job_desc = job_description
                clean_resume = resume_text
            
            # Perform comprehensive analysis
            analysis = self._analyze_candidate_fit(clean_job_desc, clean_resume, candidate_name)
            
            # Generate summary based on analysis
            summary = self._create_professional_summary(analysis, candidate_name, summary_type)
            
            return {
                'success': True,
                'summary': summary,
                'candidate_name': candidate_name,
                'analysis_details': analysis,
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
                    'analysis_method': 'enhanced_template_analysis',
                    'pii_protection_enabled': self.config.general_config['enable_pii_protection'],
                    'summary_type': summary_type or 'professional',
                    'domains_detected': analysis.get('job_domain', []),
                    'skills_matched': len(analysis.get('technical_skills', [])),
                    'experience_level': analysis.get('experience_level', 'unknown')
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Enhanced analysis error: {str(e)}",
                'candidate_name': candidate_name,
                'service_used': 'enhanced_analysis'
            }
    
    def _create_professional_summary(self, analysis: Dict[str, Any], candidate_name: str, summary_type: str = None) -> str:
        """Create a professional summary based on analysis results."""
        summary_length = self.config.prompt_config.summary_length
        
        # Extract key information from analysis
        strengths = analysis.get('strengths', [])
        technical_skills = analysis.get('technical_skills', [])
        experience_level = analysis.get('experience_level', 'experienced')
        domain_fit = analysis.get('domain_analysis', {}).get('relevance_score', 0)
        
        # Build summary components
        summary_parts = []
        
        # Opening statement
        if domain_fit > 0.7:
            summary_parts.append(f"{candidate_name} demonstrates strong alignment with this role's requirements.")
        elif domain_fit > 0.4:
            summary_parts.append(f"{candidate_name} shows relevant qualifications for this position.")
        else:
            summary_parts.append(f"{candidate_name} brings transferable skills that could benefit this role.")
        
        # Skills and experience
        if technical_skills:
            top_skills = technical_skills[:3]  # Top 3 skills
            if len(top_skills) > 1:
                skills_text = f"Key strengths include {', '.join(top_skills[:-1])}, and {top_skills[-1]}."
            else:
                skills_text = f"Demonstrates expertise in {top_skills[0]}."
            summary_parts.append(skills_text)
        
        # Experience level and fit
        if experience_level == 'senior':
            summary_parts.append("Their senior-level experience and proven track record make them well-suited for complex challenges.")
        elif experience_level == 'mid':
            summary_parts.append("Their solid professional background and demonstrated growth trajectory align well with role expectations.")
        else:
            summary_parts.append("Their foundational skills and eagerness to contribute make them a promising candidate.")
        
        # Recommendation
        if domain_fit > 0.6 and technical_skills:
            summary_parts.append("Recommend advancing to interview stage for detailed technical and cultural fit assessment.")
        else:
            summary_parts.append("Suggest further evaluation to determine specific role alignment and growth potential.")
        
        # Combine and trim to requested length
        summary = " ".join(summary_parts[:summary_length])
        
        return summary
    
    # Legacy method for backward compatibility (marked for deprecation)
    def _generate_openai_summary(self, job_description: str, resume_text: str, candidate_name: str) -> str:
        """Generate OpenAI-powered summary."""
        # Truncate texts if too long
        job_desc = job_description[:2000] + "..." if len(job_description) > 2000 else job_description
        resume = resume_text[:3000] + "..." if len(resume_text) > 3000 else resume_text
        
        prompt = f"""
Analyze how well this candidate fits the job requirements. Provide a concise, professional summary.

JOB DESCRIPTION:
{job_desc}

CANDIDATE RESUME:
{resume}

Provide a 2-3 sentence analysis covering:
1. Key qualifications match
2. Relevant experience/skills
3. Overall fit assessment

Write in a professional, positive tone. Be specific about qualifications.

SUMMARY:"""
        
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert HR analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    
    def _generate_enhanced_summary(self, job_description: str, resume_text: str, candidate_name: str) -> str:
        """Generate enhanced analysis-based summary."""
        # Comprehensive analysis
        analysis = self._analyze_candidate_fit(job_description, resume_text, candidate_name)
        
        # Generate contextual summary
        summary_parts = []
        
        # Core fit assessment
        core_fit = self._assess_core_fit(analysis)
        summary_parts.append(core_fit)
        
        # Skills analysis
        skills_analysis = self._analyze_skills_match(analysis)
        if skills_analysis:
            summary_parts.append(skills_analysis)
        
        # Experience analysis
        exp_analysis = self._analyze_experience_match(analysis)
        if exp_analysis:
            summary_parts.append(exp_analysis)
        
        # Recommendations
        recommendations = self._generate_recommendations(analysis)
        if recommendations:
            summary_parts.append(recommendations)
        
        return " ".join(summary_parts)
    
    def _analyze_candidate_fit(self, job_desc: str, resume: str, name: str) -> Dict[str, Any]:
        """Comprehensive candidate analysis."""
        job_lower = job_desc.lower()
        resume_lower = resume.lower()
        
        # Domain detection
        job_domain = self._detect_domain(job_lower)
        candidate_domains = self._detect_candidate_domains(resume_lower)
        
        # Skills extraction
        job_skills = self._extract_skills_comprehensive(job_lower)
        candidate_skills = self._extract_skills_comprehensive(resume_lower)
        
        # Experience level
        experience_level = self._extract_experience_level(resume_lower)
        required_experience = self._extract_required_experience(job_lower)
        
        # Seniority match
        job_seniority = self._detect_seniority_level(job_lower)
        candidate_seniority = self._detect_seniority_level(resume_lower)
        
        return {
            'name': name,
            'job_domain': job_domain,
            'candidate_domains': candidate_domains,
            'job_skills': job_skills,
            'candidate_skills': candidate_skills,
            'matching_skills': set(job_skills) & set(candidate_skills),
            'experience_level': experience_level,
            'required_experience': required_experience,
            'job_seniority': job_seniority,
            'candidate_seniority': candidate_seniority,
            'domain_match': job_domain in candidate_domains,
            'skill_match_ratio': len(set(job_skills) & set(candidate_skills)) / max(len(job_skills), 1)
        }
    
    def _assess_core_fit(self, analysis: Dict[str, Any]) -> str:
        """Generate core fit assessment."""
        name = analysis['name']
        skill_ratio = analysis['skill_match_ratio']
        domain_match = analysis['domain_match']
        
        if skill_ratio >= 0.7 and domain_match:
            return f"{name} presents an excellent fit for this position with exceptional alignment across core competencies and domain expertise."
        elif skill_ratio >= 0.5 and domain_match:
            return f"{name} demonstrates strong qualifications for this role with solid domain knowledge and relevant skill set."
        elif skill_ratio >= 0.3 or domain_match:
            return f"{name} shows promising potential for this position with transferable skills and relevant background."
        else:
            return f"{name} brings a unique perspective to this role with complementary skills that could add value to the team."
    
    def _analyze_skills_match(self, analysis: Dict[str, Any]) -> str:
        """Analyze skills match."""
        matching_skills = list(analysis['matching_skills'])
        
        if len(matching_skills) >= 5:
            top_skills = matching_skills[:4]
            return f"Key competencies include {', '.join(top_skills[:-1])}, and {top_skills[-1]}, demonstrating technical proficiency in critical areas."
        elif len(matching_skills) >= 2:
            return f"Shows relevant experience in {' and '.join(matching_skills)}, providing a solid foundation for the role."
        elif len(matching_skills) == 1:
            return f"Brings valuable expertise in {matching_skills[0]}."
        else:
            return ""
    
    def _analyze_experience_match(self, analysis: Dict[str, Any]) -> str:
        """Analyze experience match."""
        candidate_exp = analysis['experience_level']
        required_exp = analysis['required_experience']
        candidate_seniority = analysis['candidate_seniority']
        job_seniority = analysis['job_seniority']
        
        if candidate_exp >= required_exp and candidate_seniority == job_seniority:
            return f"Professional experience level aligns well with position requirements, demonstrating readiness for {job_seniority}-level responsibilities."
        elif candidate_exp >= required_exp:
            return f"Brings {candidate_exp}+ years of relevant experience, meeting the experience requirements for this position."
        elif candidate_exp > 0:
            return f"Has {candidate_exp} years of experience which provides a foundation for growth in this role."
        else:
            return "Represents an entry-level candidate with growth potential."
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> str:
        """Generate hiring recommendations."""
        skill_ratio = analysis['skill_match_ratio']
        domain_match = analysis['domain_match']
        
        if skill_ratio >= 0.6 and domain_match:
            return "Recommended for immediate consideration and next-round interviews."
        elif skill_ratio >= 0.4:
            return "Strong candidate worth pursuing for detailed technical assessment."
        elif skill_ratio >= 0.2:
            return "Consider for interview to explore potential and cultural fit."
        else:
            return "May benefit from additional screening to assess transferable value."
    
    def _extract_skills_comprehensive(self, text: str) -> List[str]:
        """Extract comprehensive skills."""
        skills_database = {
            # Programming Languages
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift', 'kotlin',
            # Web Technologies
            'react', 'angular', 'vue', 'node', 'express', 'django', 'flask', 'spring', 'laravel',
            # Databases
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 'oracle', 'sqlite',
            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'gitlab', 'terraform', 'ansible',
            # Data & ML
            'machine learning', 'deep learning', 'data science', 'analytics', 'tensorflow', 'pytorch', 'pandas', 'numpy',
            # Tools & Methodologies
            'git', 'agile', 'scrum', 'jira', 'confluence', 'slack', 'linux', 'windows', 'macos',
            # Business Skills
            'project management', 'leadership', 'communication', 'problem solving', 'teamwork', 'strategic planning'
        }
        
        found = []
        for skill in skills_database:
            if skill in text:
                found.append(skill)
        return found
    
    def _detect_domain(self, text: str) -> str:
        """Detect job domain."""
        for domain, keywords in self.job_domains.items():
            if any(keyword in text for keyword in keywords):
                return domain
        return 'general'
    
    def _detect_candidate_domains(self, text: str) -> List[str]:
        """Detect candidate domains."""
        domains = []
        for domain, keywords in self.job_domains.items():
            if any(keyword in text for keyword in keywords):
                domains.append(domain)
        return domains if domains else ['general']
    
    def _extract_experience_level(self, text: str) -> int:
        """Extract years of experience."""
        # Look for patterns like "5 years", "3+ years", etc.
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
            r'(\d+)\+?\s*(?:years?|yrs?)\s*experience',
            r'experience.*?(\d+)\+?\s*(?:years?|yrs?)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return max(int(match) for match in matches)
        
        return 0
    
    def _extract_required_experience(self, text: str) -> int:
        """Extract required years of experience from job description."""
        patterns = [
            r'(?:minimum|min|at least|require[sd]?)\s*(\d+)\+?\s*(?:years?|yrs?)',
            r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:minimum|min|required|experience)',
            r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:relevant\s*)?experience',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return max(int(match) for match in matches)
        
        return 0
    
    def _detect_seniority_level(self, text: str) -> str:
        """Detect seniority level."""
        senior_keywords = ['senior', 'lead', 'principal', 'staff', 'architect', 'manager', 'director']
        junior_keywords = ['junior', 'entry', 'associate', 'intern', 'trainee', 'graduate']
        
        if any(keyword in text for keyword in senior_keywords):
            return 'senior'
        elif any(keyword in text for keyword in junior_keywords):
            return 'junior'
        else:
            return 'mid'
