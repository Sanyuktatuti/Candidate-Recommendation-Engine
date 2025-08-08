"""
Unified AI service with automatic hierarchy: OpenAI → Enhanced Analysis.
"""

import os
import re
from typing import List, Dict, Any
import streamlit as st

# Load environment variables for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, that's fine
    pass

try:
    import openai
except ImportError:
    openai = None


class UnifiedAIService:
    """Unified AI service with automatic hierarchy: OpenAI → Enhanced Analysis."""
    
    def __init__(self):
        """Initialize the AI service."""
        # Priority 1: Environment variables (for local development)
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        
        # Priority 2: Streamlit secrets (for cloud deployment) - only if env var is empty
        if not self.openai_api_key:
            try:
                self.openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
            except Exception:
                # No secrets available, use empty string (will fallback to enhanced analysis)
                self.openai_api_key = ""
        
        self.openai_client = None
        if self.openai_api_key and openai:
            try:
                self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
            except Exception:
                self.openai_client = None
        
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
    
    def generate_fit_summary(self, job_description: str, resume_text: str, candidate_name: str) -> str:
        """Generate candidate fit summary using OpenAI or enhanced analysis.
        
        Args:
            job_description: The job description text
            resume_text: The candidate's resume text
            candidate_name: The candidate's name
            
        Returns:
            A professional summary of how the candidate fits the role
        """
        # Try OpenAI first for premium quality
        if self.openai_client:
            try:
                return self._generate_openai_summary(job_description, resume_text, candidate_name)
            except Exception:
                pass  # Fall back to enhanced analysis
        
        # Fallback to enhanced analysis
        try:
            return self._generate_enhanced_summary(job_description, resume_text, candidate_name)
        except Exception:
            return (f"{candidate_name} demonstrates relevant qualifications for this position. "
                   f"The profile shows professional experience that aligns with several key requirements. "
                   f"Recommend detailed interview to assess specific fit and potential contribution to the role.")
    
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
