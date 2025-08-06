"""
AI service for generating candidate fit summaries using OpenAI GPT.
"""
import asyncio
import logging
from typing import Optional
import openai
from config import settings

logger = logging.getLogger(__name__)


class AIService:
    """Service for AI-powered text generation and analysis."""
    
    def __init__(self):
        """Initialize the AI service."""
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.chat_model = settings.chat_model
        self.max_summary_length = settings.max_summary_length
    
    async def generate_fit_summary(
        self, 
        job_description: str, 
        resume_text: str, 
        candidate_name: str
    ) -> str:
        """
        Generate an AI summary explaining why a candidate is a good fit for a role.
        
        Args:
            job_description: The job description text
            resume_text: The candidate's resume text
            candidate_name: The candidate's name
            
        Returns:
            AI-generated summary explaining the fit
        """
        try:
            prompt = self._create_fit_summary_prompt(
                job_description, 
                resume_text, 
                candidate_name
            )
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.chat_model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert HR analyst specializing in candidate-job fit assessment."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_summary_length,
                temperature=0.7
            )
            
            summary = response.choices[0].message.content.strip()
            logger.info(f"Generated fit summary for candidate {candidate_name}")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating fit summary: {e}")
            return f"Unable to generate summary for {candidate_name}. Please review their qualifications manually."
    
    def _create_fit_summary_prompt(
        self, 
        job_description: str, 
        resume_text: str, 
        candidate_name: str
    ) -> str:
        """
        Create a prompt for generating candidate fit summary.
        
        Args:
            job_description: The job description
            resume_text: The candidate's resume
            candidate_name: The candidate's name
            
        Returns:
            Formatted prompt string
        """
        # Truncate texts if too long
        max_job_chars = 2000
        max_resume_chars = 3000
        
        if len(job_description) > max_job_chars:
            job_description = job_description[:max_job_chars] + "..."
        
        if len(resume_text) > max_resume_chars:
            resume_text = resume_text[:max_resume_chars] + "..."
        
        prompt = f"""
Analyze the candidate-job fit and provide a concise summary (max 150 words) explaining why {candidate_name} would be a great fit for this role.

JOB DESCRIPTION:
{job_description}

CANDIDATE RESUME:
{resume_text}

Focus on:
1. Relevant skills and experience alignment
2. Key achievements that match job requirements
3. Unique strengths that add value
4. Any potential growth areas or learning opportunities

Write in a professional, positive tone suitable for a hiring manager. Be specific about qualifications and avoid generic statements.

SUMMARY:"""
        
        return prompt
    
    async def extract_key_skills(self, text: str) -> list:
        """
        Extract key skills from a text (job description or resume).
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of extracted skills
        """
        try:
            prompt = f"""
Extract the top 10 most important skills, technologies, and qualifications mentioned in this text. 
Return them as a comma-separated list.

TEXT:
{text[:2000]}

SKILLS:"""
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.chat_model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert at identifying professional skills and qualifications from job descriptions and resumes."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            skills_text = response.choices[0].message.content.strip()
            skills = [skill.strip() for skill in skills_text.split(',')]
            return skills[:10]  # Limit to top 10
            
        except Exception as e:
            logger.error(f"Error extracting skills: {e}")
            return []
    
    async def health_check(self) -> bool:
        """
        Check if the AI service is healthy.
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            # Test with a simple completion
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.chat_model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            return len(response.choices) > 0
        except Exception as e:
            logger.error(f"AI service health check failed: {e}")
            return False


# Global AI service instance
ai_service = AIService()
