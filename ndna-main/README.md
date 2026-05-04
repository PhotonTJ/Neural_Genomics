# nDNA - Neural DNA Research Repository

A Jekyll-based academic website exploring the semantic genome of large language models and their epistemic structures.

## Overview

This repository contains research on **Neural DNA (nDNA)** - the latent semantic structures that define how AI models think, believe, and reason. The site features academic papers, visualizations, and threat analysis related to adversarial manipulation of model internals.

## Features

- **Academic Paper Conversion**: LaTeX to Markdown with MathJax support
- **Dynamic Citation System**: Auto-numbering bibliography from BibTeX files
- **Interactive Visualizations**: Enhanced figure display with captions
- **Threat Analysis Cards**: Professional UI for security research
- **FontAwesome Integration**: Consistent iconography throughout

## Prerequisites

- Ruby 3.2 (recommended: use rbenv or RVM)
- Bundler gem
- Git

## Local Development

### 1. Clone the Repository

```bash
git clone https://github.com/pragyaai/ndna.git
cd ndna
```

### 2. Install Dependencies

```bash
# Install bundler if you haven't already
gem install bundler

# Install Jekyll and dependencies
bundle install
```

### 3. Local Development Server

```bash
# Start the Jekyll development server
bundle exec jekyll serve

# Or with live reload
bundle exec jekyll serve --livereload

# Access at http://localhost:4000
```

### 4. Build for Production

```bash
# Generate static files
bundle exec jekyll build

# Output will be in _site/ directory
```

## Configuration

### Site Settings

Edit `_config.yml` for basic site configuration:

```yaml
title: nDNA Research
description: Neural DNA and AI Safety Research
url: "https://your-domain.com"
baseurl: ""
```

### Citations and Bibliography

- **See [dynamic-citations.md](dynamic-citations.md) for detailed usage examples and citation system documentation**
