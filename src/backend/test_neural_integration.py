import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Try to import from neural_integration
try:
    from src.tools.neural_integration import HAVE_TRANSFORMERS
    print(f"HAVE_TRANSFORMERS: {HAVE_TRANSFORMERS}")
    
    # If HAVE_TRANSFORMERS is False, try to import transformers directly
    if not HAVE_TRANSFORMERS:
        try:
            import transformers
            print(f"Transformers is actually available! Version: {transformers.__version__}")
        except ImportError:
            print("Confirmed: transformers is not available")
except ImportError as e:
    print(f"Error importing from neural_integration: {e}")