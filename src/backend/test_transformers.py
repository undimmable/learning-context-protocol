import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    print("Transformers is available!")
    
    # Try to import specific classes used in the project
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    print("Successfully imported AutoTokenizer and AutoModelForSeq2SeqLM")
except ImportError as e:
    print(f"Error importing transformers: {e}")
    print("Transformers is NOT available!")