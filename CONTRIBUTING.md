# Contributing to python-claw

This repository contains Python projects for OpenClaw. All code contributions must follow the Pull Request workflow.

## 🚀 Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Develop Your Code
- Write clean, documented Python code
- Add tests if applicable
- Update documentation

### 3. Commit Your Changes
```bash
git add .
git commit -m "feat: description of your change"
```

### 4. Push to GitHub
```bash
git push origin feature/your-feature-name
```

### 5. Create a Pull Request
- Go to the repository on GitHub
- Click "Compare & pull request"
- Fill in the PR template
- Request review from @netxeye

### 6. Review and Merge
- Wait for review and feedback
- Make requested changes if needed
- Once approved, @netxeye will merge the PR

## 📝 Commit Message Convention

Use conventional commit messages:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Example: `feat: add weather API integration`

## 🐍 Python Standards

- Use Python 3.8+ syntax
- Follow PEP 8 style guide
- Add type hints where appropriate
- Include docstrings for functions and classes
- Use virtual environments for dependencies

## 🏗️ Project Structure

Each project should be in its own directory:
```
python-claw/
├── projects/
│   ├── project-1/
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   └── src/
│   └── project-2/
├── CONTRIBUTING.md
└── README.md
```

## 🔧 Setup for Development

1. Clone the repository:
```bash
git clone git@github.com:netxeye/python-claw.git
cd python-claw
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies for your project:
```bash
cd projects/your-project
pip install -r requirements.txt
```

## ❓ Getting Help

If you need assistance:
1. Check existing issues and PRs
2. Create a new issue with your question
3. Tag @netxeye for urgent matters

---

**Remember:** No direct pushes to `main` branch. All changes must go through PR review.