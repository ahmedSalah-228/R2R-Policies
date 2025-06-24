import re
from r2r import R2RClient

def clean_response(text):
    """Remove reference numbers in square brackets like [6a9e83b], [261ad80]"""
    # Pattern to match square brackets containing alphanumeric characters
    pattern = r'\[[a-f0-9]+\]'
    return re.sub(pattern, '', text).strip()

client = R2RClient(base_url="http://localhost:7272")
# to ingest your own document, 
# client.documents.create(file_path="CC_Sales.pdf")
print(client.documents.list())