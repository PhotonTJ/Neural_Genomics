#!/bin/bash
# =============================================================================
# ndna Environment Setup Script
# =============================================================================
# This script:
#   1. Creates/activates a conda environment
#   2. Installs dependencies from requirements.txt and setup.py
#   3. Configures GitHub authentication
#   4. Configures HuggingFace authentication
#
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh
#
# Or to source it (keeps env activated in current shell):
#   source setup_env.sh
# =============================================================================

set -e  # Exit on first error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}=====================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=====================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# =============================================================================
# Configuration - SET THESE OR EXPORT AS ENVIRONMENT VARIABLES
# =============================================================================
GITHUB_USERNAME="${GITHUB_USERNAME:-anonymous}"
GITHUB_PAT="${GITHUB_PAT:-your_github_pat_here}"
HF_TOKEN="${HF_TOKEN:-your_hf_token_here}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ndna}"
REPO_URL="https://github.com/anonymous-submission/ndna.git"

# =============================================================================
# Step 0: Navigate to project directory
# =============================================================================
print_header "Step 0: Navigating to project directory"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# If script is in scripts/ folder, go to parent (project root)
if [[ "$(basename "$SCRIPT_DIR")" == "scripts" ]]; then
    cd "$SCRIPT_DIR/.."
else
    cd "$SCRIPT_DIR"
fi
print_success "Working directory: $(pwd)"

# =============================================================================
# Step 1: Clear VS Code Git environment variables (prevents auth conflicts)
# =============================================================================
print_header "Step 1: Clearing VS Code Git environment variables"

unset GIT_ASKPASS 2>/dev/null || true
unset SSH_ASKPASS 2>/dev/null || true
unset VSCODE_GIT_ASKPASS_NODE 2>/dev/null || true
unset VSCODE_GIT_ASKPASS_EXTRA_ARGS 2>/dev/null || true
unset VSCODE_GIT_ASKPASS_MAIN 2>/dev/null || true
unset VSCODE_GIT_IPC_HANDLE 2>/dev/null || true

print_success "VS Code Git environment variables cleared"

# =============================================================================
# Step 2: Setup Conda Environment
# =============================================================================
print_header "Step 2: Setting up Conda environment"

# Initialize conda for this shell
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]; then
    source "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
elif command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
else
    print_error "Conda not found! Please install Miniconda or Anaconda first."
    exit 1
fi

# Check if environment exists, create if not
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    print_success "Conda environment '${CONDA_ENV_NAME}' exists"
else
    print_warning "Creating conda environment '${CONDA_ENV_NAME}' with Python 3.10..."
    conda create -n "$CONDA_ENV_NAME" python=3.10 -y
    print_success "Conda environment created"
fi

# Activate environment
conda activate "$CONDA_ENV_NAME"
print_success "Activated conda environment: $CONDA_ENV_NAME"
print_success "Python location: $(which python)"
print_success "Python version: $(python --version)"

# =============================================================================
# Step 3: Install Dependencies
# =============================================================================
print_header "Step 3: Installing dependencies"

# Upgrade pip first
pip install --upgrade pip

pip install uv

# Install from setup.py if exists (editable mode)
if [ -f "setup.py" ]; then
    print_warning "Installing package in editable mode (setup.py)..."
    uv pip install -e .
    print_success "setup.py installed in editable mode"
else
    print_warning "setup.py not found, skipping"
    # Install from requirements.txt if exists
    if [ -f "requirements.txt" ]; then
        print_warning "Installing from requirements.txt..."
        uv pip install -r requirements.txt
        print_success "requirements.txt installed"
    else
        print_warning "requirements.txt not found, skipping"
    fi
fi

# =============================================================================
# Step 4: Configure GitHub Authentication
# =============================================================================
print_header "Step 4: Configuring GitHub authentication"

configure_github() {
    # Set git credentials
    git config --global user.name "$GITHUB_USERNAME"
    git config --global credential.helper store
    
    # Store credentials
    echo "https://${GITHUB_USERNAME}:${GITHUB_PAT}@github.com" > ~/.git-credentials
    chmod 600 ~/.git-credentials
    
    # Set remote URL with PAT embedded (for this repo)
    git remote set-url origin "https://${GITHUB_USERNAME}:${GITHUB_PAT}@github.com/${GITHUB_USERNAME}/ndna.git" 2>/dev/null || \
    git remote add origin "https://${GITHUB_USERNAME}:${GITHUB_PAT}@github.com/${GITHUB_USERNAME}/ndna.git" 2>/dev/null || true
    
    print_success "GitHub credentials configured"
}

test_github() {
    print_warning "Testing GitHub connection..."
    
    # Show current remote
    echo "Current remotes:"
    git remote -v
    
    # Try a git fetch to test
    if git fetch origin 2>&1; then
        print_success "GitHub connection successful!"
        return 0
    else
        print_error "GitHub connection failed"
        return 1
    fi
}

# First attempt
configure_github

if ! test_github; then
    print_warning "First attempt failed. Reconfiguring..."
    
    # Clear any cached credentials
    git credential reject <<EOF
protocol=https
host=github.com
EOF
    
    # Reset remote URL
    git remote remove origin 2>/dev/null || true
    git remote add origin "https://${GITHUB_USERNAME}:${GITHUB_PAT}@github.com/${GITHUB_USERNAME}/ndna.git"
    
    # Try again
    if ! test_github; then
        print_error "GitHub authentication failed after retry."
        print_warning "You may need to:"
        print_warning "  1. Check your PAT is valid and has repo permissions"
        print_warning "  2. Regenerate your PAT at https://github.com/settings/tokens"
        print_warning "  3. Run: git remote set-url origin https://<username>:<PAT>@github.com/${GITHUB_USERNAME}/ndna.git"
    fi
fi

# =============================================================================
# Step 5: Configure HuggingFace Authentication
# =============================================================================
print_header "Step 5: Configuring HuggingFace authentication"

# Set HF_TOKEN environment variable
export HF_TOKEN="$HF_TOKEN"

# Try to login using huggingface-cli
if command -v huggingface-cli &> /dev/null; then
    # Write token to HF cache
    mkdir -p ~/.cache/huggingface
    echo "$HF_TOKEN" > ~/.cache/huggingface/token
    chmod 600 ~/.cache/huggingface/token
    
    # Also try the CLI login (non-interactive)
    echo "$HF_TOKEN" | huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true
    
    print_success "HuggingFace token configured"
    
    # Test HF connection
    print_warning "Testing HuggingFace connection..."
    if huggingface-cli whoami 2>/dev/null; then
        print_success "HuggingFace authentication successful!"
    else
        print_warning "HuggingFace CLI test inconclusive, but token is set"
    fi
else
    print_warning "huggingface-cli not found. Installing huggingface_hub..."
    pip install huggingface_hub
    
    # Write token directly
    mkdir -p ~/.cache/huggingface
    echo "$HF_TOKEN" > ~/.cache/huggingface/token
    chmod 600 ~/.cache/huggingface/token
    
    print_success "HuggingFace token saved to ~/.cache/huggingface/token"
fi

# Also export for Python usage
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# =============================================================================
# Step 6: Verify Installation
# =============================================================================
print_header "Step 6: Verifying installation"

echo "Checking key packages..."

# Check torch
python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null && \
    print_success "PyTorch installed" || print_warning "PyTorch not installed"

# Check transformers
python -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null && \
    print_success "Transformers installed" || print_warning "Transformers not installed"

# Check peft
python -c "import peft; print(f'PEFT: {peft.__version__}')" 2>/dev/null && \
    print_success "PEFT installed" || print_warning "PEFT not installed"

# Check datasets
python -c "import datasets; print(f'Datasets: {datasets.__version__}')" 2>/dev/null && \
    print_success "Datasets installed" || print_warning "Datasets not installed"

# Check ndna_lib
python -c "import ndna_lib" 2>/dev/null && \
    print_success "ndna_lib installed" || print_warning "ndna_lib not installed (run pip install -e .)"

# =============================================================================
# Summary
# =============================================================================
print_header "Setup Complete!"

echo -e "Environment: ${GREEN}${CONDA_ENV_NAME}${NC}"
echo -e "Python: ${GREEN}$(python --version 2>&1)${NC}"
echo -e "Working directory: ${GREEN}$(pwd)${NC}"
echo ""
echo -e "${YELLOW}To activate this environment in a new terminal:${NC}"
echo -e "  ${BLUE}conda activate ${CONDA_ENV_NAME}${NC}"
echo ""
echo -e "${YELLOW}To run Method-5 generic:${NC}"
echo -e "  ${BLUE}python -m scripts.run_method5_generic --help${NC}"
echo ""
echo -e "${YELLOW}To push to GitHub:${NC}"
echo -e "  ${BLUE}git add . && git commit -m 'your message' && git push origin main${NC}"
echo ""
echo -e "${RED}SECURITY REMINDER: Regenerate your GitHub PAT and HF token after setup!${NC}"
