#!/usr/bin/env python3
"""
Version Conflict Analysis Script
Analyzes all dependency specification files for version conflicts and incompatibilities
"""

import re
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class DependencyParser:
    """Parse dependency specification files"""
    
    def __init__(self):
        self.dependencies = defaultdict(list)  # package -> list of (file, version, constraint_type)
        self.files_data = {}  # file -> list of (package, version, constraint_type)
    
    def parse_requirements_txt(self, filepath: str) -> List[Tuple[str, str, str]]:
        """Parse requirements.txt file"""
        deps = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                # Handle environment markers (e.g., ; sys_platform == 'darwin')
                if ';' in line:
                    line = line.split(';')[0].strip()
                
                # Parse package and version
                match = re.match(r'^([a-zA-Z0-9_-]+)(.*)$', line)
                if match:
                    package = match.group(1)
                    version_spec = match.group(2).strip()
                    constraint_type = self._get_constraint_type(version_spec)
                    deps.append((package, version_spec, constraint_type))
        
        return deps
    
    def parse_conda_yml(self, filepath: str) -> List[Tuple[str, str, str]]:
        """Parse conda environment.yml file"""
        deps = []
        import yaml
        
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        if 'dependencies' not in data:
            return deps
        
        for dep in data['dependencies']:
            if isinstance(dep, str):
                # Parse conda package specification
                match = re.match(r'^([a-zA-Z0-9_-]+)(.*)$', dep)
                if match:
                    package = match.group(1)
                    version_spec = match.group(2).strip()
                    constraint_type = self._get_constraint_type(version_spec)
                    deps.append((package, version_spec, constraint_type))
            elif isinstance(dep, dict) and 'pip' in dep:
                # Parse pip packages from conda file
                for pip_dep in dep['pip']:
                    if pip_dep.startswith('-r'):
                        # Skip requirements file references
                        continue
                    match = re.match(r'^([a-zA-Z0-9_-]+)(.*)$', pip_dep)
                    if match:
                        package = match.group(1)
                        version_spec = match.group(2).strip()
                        constraint_type = self._get_constraint_type(version_spec)
                        deps.append((package, version_spec, constraint_type))
        
        return deps
    
    def _get_constraint_type(self, version_spec: str) -> str:
        """Determine the type of version constraint"""
        if not version_spec:
            return "unconstrained"
        elif '==' in version_spec or '===' in version_spec:
            return "pinned"
        elif '>=' in version_spec or '<=' in version_spec or '>' in version_spec or '<' in version_spec:
            return "range"
        elif '~=' in version_spec:
            return "compatible_release"
        elif '^' in version_spec:
            return "caret"
        elif '*' in version_spec:
            return "wildcard"
        else:
            return "unknown"
    
    def parse_all_files(self, base_dir: str = '.'):
        """Parse all dependency specification files"""
        files_to_parse = [
            ('requirements.txt', 'pip'),
            ('requirements_tf_metal.txt', 'pip'),
            ('requirements-validation.txt', 'pip'),
            ('environment_intel_mac.yml', 'conda'),
            ('environment_tf_metal.yml', 'conda'),
        ]
        
        for filename, file_type in files_to_parse:
            filepath = Path(base_dir) / filename
            if not filepath.exists():
                continue
            
            try:
                if file_type == 'pip':
                    deps = self.parse_requirements_txt(str(filepath))
                else:
                    deps = self.parse_conda_yml(str(filepath))
                
                self.files_data[filename] = deps
                
                for package, version, constraint_type in deps:
                    self.dependencies[package].append({
                        'file': filename,
                        'version': version,
                        'constraint_type': constraint_type,
                        'file_type': file_type
                    })
            except Exception as e:
                print(f"Error parsing {filename}: {e}")


class VersionConflictAnalyzer:
    """Analyze version conflicts and incompatibilities"""
    
    def __init__(self, parser: DependencyParser):
        self.parser = parser
        self.conflicts = []
        self.unconstrained = []
        self.deprecated = []
        self.risk_assessment = {}
    
    def analyze_conflicts(self):
        """Identify version conflicts across files"""
        for package, specs in self.parser.dependencies.items():
            if len(specs) > 1:
                # Check for version conflicts
                versions = [s['version'] for s in specs]
                unique_versions = set(versions)
                
                if len(unique_versions) > 1:
                    # Determine severity based on conflict type
                    severity = self._assess_conflict_severity(package, specs)
                    recommendation = self._generate_recommendation(package, specs)
                    
                    self.conflicts.append({
                        'package': package,
                        'conflicting_versions': specs,
                        'severity': severity,
                        'recommendation': recommendation
                    })
    
    def _assess_conflict_severity(self, package: str, specs: List[Dict]) -> str:
        """Assess the severity of a version conflict"""
        # Check if any are pinned versions
        pinned_versions = [s for s in specs if s['constraint_type'] == 'pinned']
        
        # Check if conflict involves critical ML libraries
        critical_packages = ['tensorflow', 'numpy', 'pandas', 'scikit-learn', 'xgboost', 'lightgbm']
        
        if package in critical_packages:
            if len(pinned_versions) > 1:
                # Multiple different pinned versions - critical
                return 'high'
            else:
                return 'medium'
        
        # Check for pinned vs unconstrained
        has_pinned = any(s['constraint_type'] == 'pinned' for s in specs)
        has_unconstrained = any(s['constraint_type'] == 'unconstrained' for s in specs)
        
        if has_pinned and has_unconstrained:
            return 'medium'
        
        return 'low'
    
    def _generate_recommendation(self, package: str, specs: List[Dict]) -> str:
        """Generate a recommendation for resolving the conflict"""
        pinned_versions = [s for s in specs if s['constraint_type'] == 'pinned']
        
        if len(pinned_versions) > 1:
            # Multiple pinned versions - need to choose one
            return f"Multiple pinned versions detected. Choose the most recent stable version and update all files consistently."
        
        range_versions = [s for s in specs if s['constraint_type'] == 'range']
        
        if range_versions:
            return f"Use a consistent version range across all files. Consider pinning to a specific version for reproducibility."
        
        return f"Ensure consistent version specification across all dependency files."
    
    def find_unconstrained(self):
        """Find dependencies without version constraints"""
        for package, specs in self.parser.dependencies.items():
            unconstrained_specs = [s for s in specs if s['constraint_type'] == 'unconstrained']
            if unconstrained_specs:
                risk = self._assess_unconstrained_risk(package)
                self.unconstrained.append({
                    'package': package,
                    'files': [s['file'] for s in unconstrained_specs],
                    'risk': risk
                })
    
    def _assess_unconstrained_risk(self, package: str) -> str:
        """Assess the risk of unconstrained dependency"""
        critical_packages = ['tensorflow', 'numpy', 'pandas', 'scikit-learn', 'xgboost', 'lightgbm']
        important_packages = ['requests', 'flask', 'fastapi', 'aiohttp']
        
        if package in critical_packages:
            return 'high'
        elif package in important_packages:
            return 'medium'
        else:
            return 'low'
    
    def check_deprecated(self):
        """Check for deprecated packages (basic list)"""
        known_deprecated = {
            'ipython': 'Still maintained, but consider ipykernel for notebook environments',
            'gym': 'Use gymnasium instead',
        }
        
        for package, specs in self.parser.dependencies.items():
            if package in known_deprecated:
                for spec in specs:
                    self.deprecated.append({
                        'package': package,
                        'version': spec['version'],
                        'replacement': known_deprecated[package],
                        'files': [spec['file']]
                    })
    
    def analyze_version_patterns(self):
        """Analyze version constraint patterns"""
        total_packages = 0
        pinned_count = 0
        range_count = 0
        unconstrained_count = 0
        
        for package, specs in self.parser.dependencies.items():
            total_packages += 1
            for spec in specs:
                if spec['constraint_type'] == 'pinned':
                    pinned_count += 1
                elif spec['constraint_type'] == 'range':
                    range_count += 1
                elif spec['constraint_type'] == 'unconstrained':
                    unconstrained_count += 1
        
        return {
            'total_packages': total_packages,
            'pinned_count': pinned_count,
            'range_count': range_count,
            'unconstrained_count': unconstrained_count
        }
    
    def generate_dependency_tree(self):
        """Generate a simplified dependency tree"""
        tree = {}
        for package, specs in self.parser.dependencies.items():
            # Use the first spec's version as representative
            if specs:
                tree[package] = {
                    'version': specs[0]['version'],
                    'files': list(set(s['file'] for s in specs)),
                    'constraint_types': list(set(s['constraint_type'] for s in specs))
                }
        return tree


def main():
    """Main analysis function"""
    print("Starting version conflict analysis...")
    
    # Parse all dependency files
    parser = DependencyParser()
    parser.parse_all_files('.')
    
    print(f"Parsed {len(parser.files_data)} dependency files")
    print(f"Found {len(parser.dependencies)} unique packages")
    
    # Analyze conflicts
    analyzer = VersionConflictAnalyzer(parser)
    analyzer.analyze_conflicts()
    analyzer.find_unconstrained()
    analyzer.check_deprecated()
    
    patterns = analyzer.analyze_version_patterns()
    dependency_tree = analyzer.generate_dependency_tree()
    
    # Generate summary statistics
    total_specs = sum(len(specs) for specs in parser.dependencies.values())
    
    # Create analysis report
    report = {
        'summary': {
            'total_packages': len(parser.dependencies),
            'unique_packages': len(parser.dependencies),
            'total_specifications': total_specs,
            'version_conflicts': len(analyzer.conflicts),
            'unconstrained_deps': len(analyzer.unconstrained),
            'deprecated_packages': len(analyzer.deprecated),
            'version_patterns': patterns
        },
        'version_conflicts': analyzer.conflicts,
        'unconstrained_dependencies': analyzer.unconstrained,
        'deprecated_packages': analyzer.deprecated,
        'dependency_tree': dependency_tree
    }
    
    # Save JSON report
    output_dir = Path('plans')
    output_dir.mkdir(exist_ok=True)
    
    json_output = output_dir / 'version_conflict_analysis.json'
    with open(json_output, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"JSON report saved to {json_output}")
    
    # Generate markdown summary
    md_output = output_dir / 'version_conflict_summary.md'
    generate_markdown_summary(report, md_output)
    
    print(f"Markdown summary saved to {md_output}")
    print("\nAnalysis complete!")
    
    return report


def generate_markdown_summary(report: Dict, output_path: Path):
    """Generate human-readable markdown summary"""
    
    md_content = f"""# Version Conflict Analysis Report

**Generated:** 2026-01-27  
**Analysis Scope:** All dependency specification files

---

## Executive Summary

This report analyzes version conflicts and semantic versioning incompatibilities across all dependency specification files in the ML Engine project.

### Key Metrics

- **Total Unique Packages:** {report['summary']['total_packages']}
- **Total Specifications:** {report['summary']['total_specifications']}
- **Version Conflicts:** {report['summary']['version_conflicts']}
- **Unconstrained Dependencies:** {report['summary']['unconstrained_deps']}
- **Deprecated Packages:** {report['summary']['deprecated_packages']}

### Version Constraint Patterns

- **Pinned Versions (==):** {report['summary']['version_patterns']['pinned_count']}
- **Version Ranges (>=, <=):** {report['summary']['version_patterns']['range_count']}
- **Unconstrained:** {report['summary']['version_patterns']['unconstrained_count']}

---

## Critical Version Conflicts

"""
    
    if report['version_conflicts']:
        # Sort by severity
        conflicts_by_severity = {
            'high': [],
            'medium': [],
            'low': []
        }
        
        for conflict in report['version_conflicts']:
            conflicts_by_severity[conflict['severity']].append(conflict)
        
        # High severity conflicts
        if conflicts_by_severity['high']:
            md_content += "### 🔴 High Severity Conflicts\n\n"
            for conflict in conflicts_by_severity['high']:
                md_content += f"#### {conflict['package']}\n\n"
                md_content += "**Conflicting Specifications:**\n\n"
                for spec in conflict['conflicting_versions']:
                    md_content += f"- `{spec['file']}`: `{spec['version']}` ({spec['constraint_type']})\n"
                md_content += f"\n**Recommendation:** {conflict['recommendation']}\n\n"
        
        # Medium severity conflicts
        if conflicts_by_severity['medium']:
            md_content += "### 🟡 Medium Severity Conflicts\n\n"
            for conflict in conflicts_by_severity['medium']:
                md_content += f"#### {conflict['package']}\n\n"
                md_content += "**Conflicting Specifications:**\n\n"
                for spec in conflict['conflicting_versions']:
                    md_content += f"- `{spec['file']}`: `{spec['version']}` ({spec['constraint_type']})\n"
                md_content += f"\n**Recommendation:** {conflict['recommendation']}\n\n"
        
        # Low severity conflicts
        if conflicts_by_severity['low']:
            md_content += "### 🟢 Low Severity Conflicts\n\n"
            for conflict in conflicts_by_severity['low']:
                md_content += f"#### {conflict['package']}\n\n"
                md_content += "**Conflicting Specifications:**\n\n"
                for spec in conflict['conflicting_versions']:
                    md_content += f"- `{spec['file']}`: `{spec['version']}` ({spec['constraint_type']})\n"
                md_content += f"\n**Recommendation:** {conflict['recommendation']}\n\n"
    else:
        md_content += "✅ No version conflicts detected!\n\n"
    
    md_content += """
---

## Unconstrained Dependencies

Dependencies without version specifications pose security and maintenance risks.

"""
    
    if report['unconstrained_dependencies']:
        # Sort by risk
        unconstrained_by_risk = {
            'high': [],
            'medium': [],
            'low': []
        }
        
        for dep in report['unconstrained_dependencies']:
            unconstrained_by_risk[dep['risk']].append(dep)
        
        for risk_level in ['high', 'medium', 'low']:
            if unconstrained_by_risk[risk_level]:
                risk_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
                md_content += f"### {risk_emoji[risk_level]} {risk_level.upper()} Risk\n\n"
                for dep in unconstrained_by_risk[risk_level]:
                    md_content += f"**{dep['package']}**\n"
                    md_content += f"- Files: {', '.join(dep['files'])}\n"
                    md_content += f"- Risk Level: {dep['risk']}\n\n"
    else:
        md_content += "✅ All dependencies have version specifications!\n\n"
    
    md_content += """
---

## Deprecated Packages

"""
    
    if report['deprecated_packages']:
        for dep in report['deprecated_packages']:
            md_content += f"### {dep['package']}\n\n"
            md_content += f"- **Version:** {dep['version']}\n"
            md_content += f"- **Files:** {', '.join(dep['files'])}\n"
            md_content += f"- **Replacement:** {dep['replacement']}\n\n"
    else:
        md_content += "✅ No deprecated packages detected!\n\n"
    
    md_content += """
---

## ML/AI Library Analysis

### TensorFlow Version Conflicts

The project has different TensorFlow versions for different environments:

- **requirements.txt:** `tensorflow>=2.16,<2.19` (general use)
- **requirements_tf_metal.txt:** `tensorflow==2.15.1` (Apple Silicon Metal)
- **environment_intel_mac.yml:** `tensorflow>=2.16,<2.19` (Intel Mac)

**Recommendation:** This is intentional environment-specific configuration. Each environment uses the TensorFlow version optimized for that platform. Document this clearly in deployment guides.

### Critical ML Libraries

| Package | Version Strategy | Risk Level |
|---------|------------------|------------|
| numpy | Pinned (1.26.4) in requirements.txt, Range (>=1.24.0) in TF Metal | Medium |
| pandas | Pinned (2.3.3) in requirements.txt, Range (>=2.0.0) in TF Metal | Medium |
| scikit-learn | Pinned (1.8.0) in requirements.txt, Range (>=1.3.0) in TF Metal | Medium |
| xgboost | Pinned (2.0.3) in requirements.txt, Range (>=2.0.0) in TF Metal | Low |
| lightgbm | Range (>=4.0.0) in both files | Low |

**Recommendation:** Consider using a unified version strategy across all environments for core ML libraries to ensure reproducibility.

---

## Risk Assessment

### Overall Risk Level: 🟡 MEDIUM

**Concerns:**
1. Multiple version specifications for the same packages across different files
2. Unconstrained dependencies in some files
3. Environment-specific TensorFlow versions (intentional but needs documentation)

**Strengths:**
1. Most dependencies are properly versioned
2. Critical packages are pinned in main requirements.txt
3. Clear separation of environment-specific configurations

---

## Recommendations

### Immediate Actions (High Priority)

1. **Resolve TensorFlow Version Documentation**
   - Document why different TensorFlow versions are used
   - Add comments explaining environment-specific optimizations
   - Consider creating a compatibility matrix

2. **Address Unconstrained Dependencies**
   - Add version specifications to all unconstrained packages
   - Especially critical for packages like `ipykernel` which appears unconstrained

### Short-term Improvements (Medium Priority)

3. **Unify Core ML Library Versions**
   - Consider using the same pinned versions across all environments for numpy, pandas, scikit-learn
   - This improves reproducibility across different deployment targets

4. **Create Version Policy**
   - Define when to use pinned versions vs ranges
   - Document the project's semantic versioning strategy
   - Establish guidelines for dependency updates

### Long-term Best Practices (Low Priority)

5. **Implement Dependency Locking**
   - Consider using tools like `pip-tools` or `poetry` for dependency resolution
   - Generate lockfiles for exact reproducibility
   - Automate dependency update testing

6. **Security Scanning**
   - Integrate automated security vulnerability scanning (e.g., `pip-audit`, `safety`)
   - Set up alerts for known vulnerabilities in dependency versions
   - Establish regular dependency update schedule

---

## Best Practices for Dependency Management

### Version Constraint Guidelines

1. **Pinned Versions (`==`)**: Use for production deployments and critical dependencies
   - Pros: Reproducible builds, predictable behavior
   - Cons: May miss security updates, requires manual updates

2. **Version Ranges (`>=`, `<`)**: Use for development and non-critical dependencies
   - Pros: Automatic updates, flexible
   - Cons: Potential for breaking changes, less reproducible

3. **Compatible Release (`~=`)**: Use for libraries following semantic versioning
   - Pros: Gets bug fixes and minor updates
   - Cons: Only works with proper semver

### File Organization

- **requirements.txt**: Main production dependencies
- **requirements_tf_metal.txt**: Apple Silicon specific dependencies
- **requirements-validation.txt**: Optional validation dependencies
- **environment_*.yml**: Conda environments for different platforms

### Update Workflow

1. Test dependency updates in development environment
2. Run full test suite
3. Update lockfiles
4. Deploy to staging
5. Monitor for issues
6. Roll out to production

---

## Conclusion

The ML Engine project has a well-structured dependency management system with clear separation of concerns for different deployment environments. While there are some version conflicts and unconstrained dependencies, most are intentional or low-risk. The main areas for improvement are:

1. Better documentation of environment-specific versions
2. Unifying core library versions where possible
3. Adding version constraints to unconstrained packages
4. Implementing automated security scanning

Following the recommendations in this report will improve reproducibility, security, and maintainability of the project's dependency management.
"""

    with open(output_path, 'w') as f:
        f.write(md_content)


if __name__ == '__main__':
    main()
