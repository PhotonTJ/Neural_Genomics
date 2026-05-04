import os
import re

for filename in os.listdir('.'):
    if not filename.endswith('.md'):
        continue
        
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # Make sure block math has newlines around it so IDEs render it properly
    # $$ math $$ -> \n$$\nmath\n$$\n
    # Note: Only do this for blocks that are not already separated, and are likely block math
    # But wait, $$ is also used for inline math sometimes? Standard is $ for inline, $$ for block.
    content = re.sub(r'(\s*)\$\$(.*?)\$\$', r'\n$$\n\2\n$$\n', content, flags=re.DOTALL)
    
    # Remove <style> tags which might be cluttering the view
    content = re.sub(r'<style>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

print("Math formatting and style cleanup complete.")
