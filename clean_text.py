import re

def preprocess_text(text, company_name=None):
    """Clean and normalize contract text"""
    text = text.lower()  # Convert to lowercase
    
    # Remove Roman numerals and indicators like (ii)
    text = re.sub(r'\b(i{1,3}|iv|v|vi{0,3}|ix|x)\b|\(\s*i{1,3}\s*\)', ' ', text)
    
    # Remove emails
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+|www\.\S+', ' ', text)
    
    # Remove physical addresses (rough)
    text = re.sub(
        r'\b(?:p\.?\s?o\.?\s?box|suite|floor|building|road|avenue|st\.?|street|zip|zipcode|city|state|country)\b[\w\s,.]*',
        ' ', text)
    
    # Remove phone numbers
    text = re.sub(r'\+?\d[\d\s\-\(\)]{7,}\d', ' ', text)
    
    # Remove section numbers like 1.1, 2), etc.
    text = re.sub(r'\b\d+(?:\.\d+)*[.)]?', ' ', text)
    
    # Remove a), b. etc.
    text = re.sub(r'\b[a-z][.)]', ' ', text)
    
    # Remove all digits
    text = re.sub(r'\d+', ' ', text)
    
    # Remove emojis and all non-alphabetic characters (keep spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove isolated single letters
    text = re.sub(r'\b[a-z]\b', ' ', text)

    # Remove possessives like "company's"
    text = re.sub(r"\b\w+'s\b", '', text)

    if company_name:
        base = re.escape(company_name.lower())
        variants = [
            base,
            base.replace('-', ' '),
            base.replace(' ', ''),
            base.replace('.', ' ')
        ]
        for variant in variants:
            text = re.sub(r'\b' + variant + r'\b', ' ', text)
    return text