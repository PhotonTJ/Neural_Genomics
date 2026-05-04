# Dynamic Auto-Citations from BibTeX

You can dynamically cite papers from your `references.bib` file! The system now automatically generates numbered citations and reference lists.

## 🎯 How It Works

When you use `{% cite wallace2019universal %}` anywhere in your text, it automatically:
1. **Generates a numbered link** like [[1]](#ref1)
2. **Tracks the citation** for this page
3. **Will appear** in the auto-generated reference list below

## 🔢 Auto-Numbering Examples

**Individual citations with auto-numbering:**
- Here's the first citation: {% cite wallace2019universal %}
- Second citation: {% cite carlini2022quantifying %}
- Third citation: {% cite goodfellow2014explaining %}

**Multiple citations in text:**
Recent work on adversarial triggers {% cite wallace2019universal %} has shown that language models can be vulnerable to specific input patterns. This builds on earlier work on adversarial examples {% cite goodfellow2014explaining %} and quantifying robustness {% cite carlini2022quantifying %}.

## 📚 Usage Modes

### 1. **Auto-Citations** (Recommended)
```liquid
{% cite wallace2019universal %}  → [[1]](#ref1)
{% cite carlini2022quantifying %} → [[2]](#ref2)
{% auto_references %}             → Full reference list
```

### 2. **Manual References** (Legacy)
```liquid
{% references wallace2019universal,carlini2022quantifying %}
```

### 3. **Mixed Mode**
You can use both on the same page if needed!

## ✨ Benefits

✅ **Auto-numbering** - Citations number themselves sequentially  
✅ **No manual tracking** - Reference list generates automatically  
✅ **Consistent formatting** - All citations follow the same style  
✅ **Dynamic updates** - Change the bib file, citations update automatically  
✅ **Error handling** - Missing citations show the key instead of breaking  
✅ **Performance** - Bib file is parsed once and cached per page  

## 🔗 Auto-Generated Reference List

The reference list below is automatically generated from all the `{% cite %}` tags used on this page:

{% auto_references %}
