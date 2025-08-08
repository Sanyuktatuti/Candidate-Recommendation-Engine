"""
Document processing service for extracting text from various file formats.
"""

import io
from typing import Optional
import streamlit as st

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import docx
except ImportError:
    docx = None


class DocumentProcessor:
    """Service for processing different document types and extracting text."""
    
    @staticmethod
    def extract_text_from_pdf(content: bytes) -> str:
        """Extract text from PDF content.
        
        Args:
            content: PDF file content as bytes
            
        Returns:
            Extracted text or empty string if extraction fails
        """
        if PyPDF2 is None:
            st.error("PyPDF2 not installed. Cannot process PDF files.")
            return ""
            
        try:
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_parts = []
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text.strip():
                    text_parts.append(page_text)
                    
            return '\n'.join(text_parts)
            
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_docx(content: bytes) -> str:
        """Extract text from DOCX content.
        
        Args:
            content: DOCX file content as bytes
            
        Returns:
            Extracted text or empty string if extraction fails
        """
        if docx is None:
            st.error("python-docx not installed. Cannot process DOCX files.")
            return ""
            
        try:
            docx_file = io.BytesIO(content)
            doc = docx.Document(docx_file)
            text_parts = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text)
                    
            return '\n'.join(text_parts)
            
        except Exception as e:
            st.error(f"Error reading DOCX: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_txt(content: bytes) -> str:
        """Extract text from TXT content.
        
        Args:
            content: TXT file content as bytes
            
        Returns:
            Decoded text or empty string if decoding fails
        """
        try:
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    return content.decode(encoding)
                except UnicodeDecodeError:
                    continue
                    
            return ""
            
        except Exception as e:
            st.error(f"Error reading TXT: {e}")
            return ""
    
    def process_uploaded_file(self, uploaded_file) -> str:
        """Process an uploaded file and extract text.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Extracted text content
        """
        if not uploaded_file:
            return ""
        
        content = uploaded_file.read()
        filename = uploaded_file.name.lower()
        
        if filename.endswith('.pdf'):
            return self.extract_text_from_pdf(content)
        elif filename.endswith('.docx'):
            return self.extract_text_from_docx(content)
        elif filename.endswith('.txt'):
            return self.extract_text_from_txt(content)
        else:
            st.error(f"Unsupported file type: {filename}")
            return ""
    
    def get_supported_extensions(self) -> list:
        """Get list of supported file extensions.
        
        Returns:
            List of supported file extensions
        """
        return ['pdf', 'docx', 'txt']
