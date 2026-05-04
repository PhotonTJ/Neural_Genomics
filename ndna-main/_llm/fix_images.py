import os
import re

replacements = {
    r'../assets/gifs/alignment/africa_ndna_final\.gif': '../assets/gifs/FINE-TUNING/LLAMA_CULTURSHIFTFINAL/llama_Africa_rotation.gif',
    r'../assets/gifs/alignment/asia_ndna_collapse\.gif': '../assets/gifs/FINE-TUNING/LLAMA_CULTURSHIFTFINAL/llama_Asia_rotation.gif',
    r'../assets/gifs/alignment/china_ndna_final\.gif': '../assets/gifs/FINE-TUNING/LLAMA_CULTURSHIFTFINAL/llama_China_rotation.gif',
    r'../assets/gifs/alignment/europe_ndna_collapse_FINAL\.gif': '../assets/gifs/FINE-TUNING/LLAMA_CULTURSHIFTFINAL/llama_Europe_rotation.gif',
    r'../assets/gifs/alignment/latinamerica\.gif': '../assets/gifs/FINE-TUNING/LLAMA_CULTURSHIFTFINAL/llama_LatinAmerica_rotation.gif',
    r'../assets/gifs/alignment/middleeast_ndna_final\.gif': '../assets/gifs/FINE-TUNING/LLAMA_CULTURSHIFTFINAL/llama_MiddleEast_rotation.gif',
    r'../assets/gifs/alignment/northamerica_ndna_collapse_FINAL\.gif': '../assets/gifs/FINE-TUNING/LLAMA_CULTURSHIFTFINAL/llama_NorthAmerica_rotation.gif',
    r'../assets/gifs/alignment/llama_collapse_v2_1\.gif': '../assets/gifs/llama_vs_cultures_offspring_students.gif',
}

for filename in os.listdir('.'):
    if not filename.endswith('.md'):
        continue
        
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    for old_pattern, new_path in replacements.items():
        content = re.sub(old_pattern, new_path, content)
        
    if content != original_content:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)

print("Image paths updated.")
