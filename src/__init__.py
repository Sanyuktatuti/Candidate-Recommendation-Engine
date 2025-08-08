"""
Candidate Recommendation Engine - Modular Architecture

A clean, maintainable implementation of an AI-powered candidate recommendation system
with automatic service hierarchy and professional UI components.
"""

__version__ = "2.0.0"
__author__ = "Candidate Recommendation Engine Team"

# Core modules
from . import models
from . import services  
from . import utils
from . import ui

__all__ = ["models", "services", "utils", "ui"]
