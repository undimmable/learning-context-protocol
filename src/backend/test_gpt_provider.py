import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Try to create a GPTLikeModelProvider and check its tokenizer
try:
    from src.tools.neural_integration import GPTLikeModelProvider, HAVE_TRANSFORMERS
    print(f"HAVE_TRANSFORMERS: {HAVE_TRANSFORMERS}")
    
    # Create a provider with default settings
    provider = GPTLikeModelProvider()
    
    # Check what type of tokenizer is being used
    print(f"Tokenizer type: {type(provider.tokenizer).__name__}")
    
    # Check if it's a HuggingFace tokenizer
    if hasattr(provider.tokenizer, 'vocab_size'):
        print(f"Tokenizer vocab size: {provider.tokenizer.vocab_size}")
    
    # Try to encode some text
    text = "Hello, world!"
    encoded = provider.tokenizer.encode(text)
    print(f"Encoded text: {encoded[:10]}... (length: {len(encoded)})")
    
    # Try to decode the encoded text
    decoded = provider.tokenizer.decode(encoded)
    print(f"Decoded text: {decoded}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()