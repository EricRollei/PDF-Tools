# Update sam2_scripts/__init__.py:

"""
SAM2 Scripts Package
"""

# Import local modules with error handling
try:
    from .sam2_integration import SAM2FlorenceIntegration
    SAM2_INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ SAM2 integration not available: {e}")
    SAM2_INTEGRATION_AVAILABLE = False

# Export what's available
__all__ = []

if SAM2_INTEGRATION_AVAILABLE:
    __all__.extend(['SAM2FlorenceIntegration'])

__all__.extend(['SAM2_INTEGRATION_AVAILABLE'])

def get_sam2_analyzer():
    """Get SAM2Florence integration"""
    if SAM2_INTEGRATION_AVAILABLE:
        return SAM2FlorenceIntegration()
    return None

__all__.append('get_sam2_analyzer')