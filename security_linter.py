"""
Security Linter
Scans code for potential secrets, passwords, tokens, and sensitive information.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import argparse


class SecretDetector:
    """Detects potential secrets and sensitive information in code."""
    
    # Common patterns for secrets
    PATTERNS = {
        'api_key': [
            r'api[_-]?key\s*[:=]\s*["\']([^"\']+)["\']',
            r'apikey\s*[:=]\s*["\']([^"\']+)["\']',
            r'["\']([a-zA-Z0-9]{32,})["\']\s*#\s*api[_-]?key',
        ],
        'password': [
            r'password\s*[:=]\s*["\']([^"\']+)["\']',
            r'pwd\s*[:=]\s*["\']([^"\']+)["\']',
            r'passwd\s*[:=]\s*["\']([^"\']+)["\']',
            r'["\']([^"\']{8,})["\']\s*#\s*password',
        ],
        'token': [
            r'token\s*[:=]\s*["\']([^"\']+)["\']',
            r'access[_-]?token\s*[:=]\s*["\']([^"\']+)["\']',
            r'bearer[_-]?token\s*[:=]\s*["\']([^"\']+)["\']',
            r'["\']([a-zA-Z0-9]{20,})["\']\s*#\s*token',
        ],
        'secret_key': [
            r'secret[_-]?key\s*[:=]\s*["\']([^"\']+)["\']',
            r'secret\s*[:=]\s*["\']([^"\']+)["\']',
            r'private[_-]?key\s*[:=]\s*["\']([^"\']+)["\']',
        ],
        'aws_key': [
            r'aws[_-]?access[_-]?key[_-]?id\s*[:=]\s*["\']([^"\']+)["\']',
            r'aws[_-]?secret[_-]?access[_-]?key\s*[:=]\s*["\']([^"\']+)["\']',
        ],
        'github_token': [
            r'github[_-]?token\s*[:=]\s*["\']([^"\']+)["\']',
            r'ghp_[a-zA-Z0-9]{36}',
        ],
        'private_key': [
            r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----',
            r'-----BEGIN\s+DSA\s+PRIVATE\s+KEY-----',
            r'-----BEGIN\s+EC\s+PRIVATE\s+KEY-----',
        ],
        'database_url': [
            r'postgresql://[^"\'\s]+',
            r'mysql://[^"\'\s]+',
            r'mongodb://[^"\'\s]+',
            r'database[_-]?url\s*[:=]\s*["\']([^"\']+)["\']',
        ],
        'email_password': [
            r'email[_-]?password\s*[:=]\s*["\']([^"\']+)["\']',
            r'smtp[_-]?password\s*[:=]\s*["\']([^"\']+)["\']',
        ],
    }
    
    # Patterns that are likely false positives (common words/variables)
    FALSE_POSITIVE_PATTERNS = [
        r'password\s*=\s*["\']\s*["\']',  # Empty password
        r'password\s*=\s*None',
        r'password\s*=\s*""',
        r'#\s*password',  # Comment about password
        r'password\s*:\s*str',  # Type annotation
        r'password\s*:\s*Optional\[str\]',  # Type annotation
    ]
    
    def __init__(self):
        self.compiled_patterns = {
            category: [re.compile(pattern, re.IGNORECASE) 
                      for pattern in patterns]
            for category, patterns in self.PATTERNS.items()
        }
        self.false_positive_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.FALSE_POSITIVE_PATTERNS
        ]
    
    def scan_file(self, file_path: Path) -> List[Tuple[int, str, str, str]]:
        """
        Scan a file for potential secrets.
        
        Returns:
            List of tuples: (line_number, category, matched_text, context)
        """
        issues = []
        
        # Skip scanning the linter itself
        if file_path.name == 'security_linter.py':
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            return [(0, 'error', f'Could not read file: {e}', '')]
        
        for line_num, line in enumerate(lines, 1):
            # Skip regex pattern definitions (common false positives)
            if 'r\'' in line or 'r"' in line or 're.compile' in line or 're.search' in line or 're.match' in line:
                # Check if this looks like a regex pattern definition
                if 'PATTERNS' in line or 'pattern' in line.lower() or 'compile' in line.lower():
                    continue
            
            # Check for false positives first
            is_false_positive = any(
                pattern.search(line) for pattern in self.false_positive_patterns
            )
            if is_false_positive:
                continue
            
            # Check each category
            for category, patterns in self.compiled_patterns.items():
                for pattern in patterns:
                    matches = pattern.finditer(line)
                    for match in matches:
                        # Extract matched text - prefer captured group if available, else full match
                        if match.groups() and match.group(1):
                            matched_text = match.group(1)
                        else:
                            matched_text = match.group(0)
                        
                        # Skip if it looks like a placeholder or example
                        if self._is_placeholder(matched_text):
                            continue
                        
                        # Skip regex patterns (they contain brackets and special chars)
                        if any(char in matched_text for char in ['[^', '\\', '(', ')', '{', '}']):
                            continue
                        
                        # Get context (surrounding code)
                        context = line.strip()[:100]  # First 100 chars
                        issues.append((line_num, category, matched_text, context))
        
        return issues
    
    def _is_placeholder(self, text: str) -> bool:
        """Check if the matched text is likely a placeholder or example."""
        placeholder_indicators = [
            'your_', 'example_', 'placeholder', 'xxxx', '****',
            'changeme', 'replace', 'todo', 'fixme', 'example.com',
            'test', 'dummy', 'sample', 'demo'
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in placeholder_indicators)
    
    def scan_directory(self, directory: Path, extensions: List[str] = None) -> Dict[str, List]:
        """
        Scan a directory for potential secrets.
        
        Args:
            directory: Directory to scan
            extensions: List of file extensions to scan (default: .py, .txt, .md, .yml, .yaml, .env)
        
        Returns:
            Dictionary mapping file paths to lists of issues
        """
        if extensions is None:
            extensions = ['.py', '.txt', '.md', '.yml', '.yaml', '.env', '.json', '.toml']
        
        results = {}
        
        # Files/directories to ignore
        ignore_dirs = {'.git', '__pycache__', 'venv', 'env', '.venv', 'node_modules'}
        ignore_files = {'.gitignore', '.gitattributes', 'security_linter.py'}  # Don't scan the linter itself
        
        for file_path in directory.rglob('*'):
            # Skip ignored directories
            if any(ignore_dir in file_path.parts for ignore_dir in ignore_dirs):
                continue
            
            # Skip ignored files
            if file_path.name in ignore_files:
                continue
            
            # Only scan files with specified extensions
            if file_path.is_file() and file_path.suffix in extensions:
                issues = self.scan_file(file_path)
                if issues:
                    results[str(file_path)] = issues
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Security linter to detect secrets and sensitive information'
    )
    parser.add_argument(
        'paths',
        nargs='*',
        default=['.'],
        help='Files or directories to scan (default: current directory)'
    )
    parser.add_argument(
        '--exit-error',
        action='store_true',
        help='Exit with error code if secrets are found'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output'
    )
    
    args = parser.parse_args()
    
    detector = SecretDetector()
    all_issues = {}
    
    for path_str in args.paths:
        path = Path(path_str)
        
        if not path.exists():
            print(f"⚠️  Path does not exist: {path}", file=sys.stderr)
            continue
        
        if path.is_file():
            issues = detector.scan_file(path)
            if issues:
                all_issues[str(path)] = issues
        elif path.is_dir():
            dir_issues = detector.scan_directory(path)
            all_issues.update(dir_issues)
    
    # Report findings
    if not all_issues:
        print("✅ No potential secrets detected!")
        return 0
    
    print("🚨 POTENTIAL SECRETS DETECTED!")
    print("=" * 80)
    
    total_issues = sum(len(issues) for issues in all_issues.values())
    
    for file_path, issues in sorted(all_issues.items()):
        print(f"\n📄 {file_path}")
        print("-" * 80)
        
        for line_num, category, matched_text, context in issues:
            # Truncate matched text if too long
            display_text = matched_text[:50] + "..." if len(matched_text) > 50 else matched_text
            
            print(f"  Line {line_num}: [{category.upper()}]")
            print(f"    Matched: {display_text}")
            if args.verbose:
                print(f"    Context: {context}")
            print()
    
    print("=" * 80)
    print(f"⚠️  Found {total_issues} potential secret(s) in {len(all_issues)} file(s)")
    print("\n💡 TIP: Never commit secrets to version control!")
    print("   Use environment variables or secret management tools instead.")
    
    return 1 if args.exit_error else 0


if __name__ == '__main__':
    sys.exit(main())

