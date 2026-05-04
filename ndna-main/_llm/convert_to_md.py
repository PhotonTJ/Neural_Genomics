import os
import re

for filename in os.listdir('.'):
    if not filename.endswith('.md'):
        continue
        
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Remove Jekyll front matter
    content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)
    
    # Remove MathJax configuration scripts
    content = re.sub(r'<!-- MathJax.*?-->', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
    
    # Replace ndna-title liquid tag with H1
    def repl_title(m):
        return f"# {m.group(1)}\n"
    content = re.sub(r'\{%\s*include\s+ndna-title\.liquid.*?title="([^"]+)".*?%\}', repl_title, content, flags=re.DOTALL)
    
    # Replace youtube video include
    def repl_video(m):
        return f"\n[Watch Inspiration Video]({m.group(1)})\n"
    content = re.sub(r'\{%\s*include\s+inspiration-video\.liquid.*?youtube_url="([^"]+)".*?%\}', repl_video, content, flags=re.DOTALL)
    
    # Replace visualization-html liquid tag with markdown image
    def repl_viz(m):
        img_path = m.group(1)
        if img_path.startswith('gifs/'):
            img_path = '../assets/' + img_path
        elif not img_path.startswith('../') and not img_path.startswith('http'):
            img_path = '../assets/' + img_path
        
        return f"![Visualization]({img_path})\n"
    content = re.sub(r'\{%\s*include\s+visualization-html\.liquid.*?image_path="([^"]+)".*?%\}', repl_viz, content, flags=re.DOTALL)
    
    # Handle {{ variable | markdownify }}
    content = re.sub(r'\{\{\s*([a-zA-Z0-9_]+)\s*\|\s*markdownify\s*\}\}', r'*(Refer to caption: \1)*', content)
    content = re.sub(r'\{\{.*?\}\}', '*(Caption)*', content)
    
    # Remove any leftover {% ... %} tags
    content = re.sub(r'\{%.*?%\}', '', content, flags=re.DOTALL)
    
    # Math conversions for GitHub markdown
    content = re.sub(r'\\\((.*?)\\\)', r'$\1$', content, flags=re.DOTALL)
    content = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', content, flags=re.DOTALL)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

print("Conversion complete.")
