#!/usr/bin/env python3
"""
Circular Dependency Detection Script

This script detects and documents all circular dependencies in the codebase
using DFS-based cycle detection algorithms.
"""

import json
from typing import Dict, List
from collections import defaultdict


def load_dependency_tree(file_path: str) -> Dict:
    """Load the dependency tree from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def load_import_data(file_path: str) -> Dict:
    """Load the import data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def build_import_graph(import_data: Dict) -> Dict[str, List[str]]:
    """
    Build a directed graph of local imports.
    
    Returns a dictionary where keys are modules and values are lists of modules they import.
    """
    graph = defaultdict(list)
    
    for file_path, file_info in import_data.get('files', {}).items():
        for imp in file_info.get('imports', []):
            # Only consider local imports
            if imp.get('type') == 'local':
                from_module = file_path
                to_module = imp.get('module')
                
                # Normalize module names to match dependency tree format
                # Convert file paths to module names
                from_module_normalized = from_module.replace('/', '.').replace('.py', '')
                
                # Add edge to graph
                graph[from_module_normalized].append(to_module)
    
    return dict(graph)


def detect_cycles_dfs(graph: Dict[str, List[str]]) -> List[List[str]]:
    """
    Detect all cycles in the directed graph using DFS.
    
    Returns a list of cycles, where each cycle is a list of module names.
    """
    cycles = []
    visited = set()
    rec_stack = set()
    
    def dfs(node: str, path: List[str]):
        """DFS traversal to detect cycles."""
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor, path.copy())
            elif neighbor in rec_stack:
                # Found a cycle
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                cycles.append(cycle)
        
        rec_stack.remove(node)
    
    for node in graph:
        if node not in visited:
            dfs(node, [])
    
    # Remove duplicate cycles (same cycle in different order)
    unique_cycles = []
    seen_cycles = set()
    
    for cycle in cycles:
        # Normalize cycle representation
        cycle_tuple = tuple(cycle)
        # Check all rotations of the cycle
        is_duplicate = False
        for i in range(len(cycle)):
            rotated = tuple(cycle[i:] + cycle[:i])
            if rotated in seen_cycles:
                is_duplicate = True
                break
        if not is_duplicate:
            seen_cycles.add(cycle_tuple)
            unique_cycles.append(cycle)
    
    return unique_cycles


def classify_cycle(cycle: List[str]) -> str:
    """
    Classify a cycle by type.
    
    Returns: 'direct', 'indirect', or 'self_referential'
    """
    if len(cycle) == 2:
        # A -> B -> A
        return 'direct'
    elif len(cycle) == 1:
        # A -> A (self-referential)
        return 'self_referential'
    else:
        # A -> B -> C -> ... -> A (3+ modules)
        return 'indirect'


def assess_severity(cycle: List[str], dependency_tree: Dict, import_data: Dict) -> str:
    """
    Assess the severity of a circular dependency.
    
    Returns: 'critical', 'high', 'medium', or 'low'
    """
    cycle_length = len(cycle)
    
    # Get module importance from dependency tree
    modules_info = dependency_tree.get('modules', {})
    
    # Count how many modules in the cycle are hub modules (highly imported)
    hub_count = 0
    for module in cycle:
        if module in modules_info:
            num_importers = modules_info[module].get('num_importers', 0)
            if num_importers >= 5:  # Hub module threshold
                hub_count += 1
    
    # Severity assessment logic
    if cycle_length == 1:
        # Self-referential - always critical
        return 'critical'
    elif cycle_length == 2:
        # Direct cycle - very problematic
        if hub_count >= 2:
            return 'critical'
        elif hub_count >= 1:
            return 'high'
        else:
            return 'medium'
    else:
        # Indirect cycle
        if cycle_length == 3 and hub_count >= 2:
            return 'critical'
        elif cycle_length <= 3 and hub_count >= 1:
            return 'high'
        elif cycle_length <= 4:
            return 'medium'
        else:
            return 'low'


def get_import_line_numbers(import_data: Dict, from_module: str, to_module: str) -> List[int]:
    """Get line numbers where a specific import occurs."""
    lines = []
    
    # Normalize module name
    from_module_file = from_module.replace('.', '/')
    
    # Find the file in import data
    for file_path, file_info in import_data.get('files', {}).items():
        if file_path == from_module_file or file_path == from_module_file + '.py':
            for imp in file_info.get('imports', []):
                if imp.get('type') == 'local' and imp.get('module') == to_module:
                    lines.append(imp.get('line', 0))
    
    return lines


def generate_recommendation(cycle_type: str, severity: str, cycle: List[str]) -> str:
    """Generate a recommendation for fixing a circular dependency."""
    if cycle_type == 'self_referential':
        return "Module imports itself. Remove the self-referential import or refactor the code structure."
    elif cycle_type == 'direct':
        return f"Direct circular import between {cycle[0]} and {cycle[1]}. Consider extracting common functionality into a separate module, using dependency injection, or reorganizing the code structure."
    else:
        return f"Indirect circular dependency involving {len(cycle)} modules. Consider: 1) Extracting shared code into a new module, 2) Using lazy imports, 3) Refactoring to break the cycle, or 4) Using interfaces/abstract base classes."


def analyze_impact(cycle: List[str], dependency_tree: Dict) -> Dict[str, str]:
    """Analyze the impact of a circular dependency."""
    modules_info = dependency_tree.get('modules', {})
    
    # Count total importers for all modules in cycle
    total_importers = sum(
        modules_info.get(m, {}).get('num_importers', 0) 
        for m in cycle
    )
    
    # Determine impact levels
    if total_importers > 20:
        runtime_risk = 'high'
        testing_impact = 'high'
        maintainability = 'high'
        refactoring_difficulty = 'high'
    elif total_importers > 10:
        runtime_risk = 'medium'
        testing_impact = 'high'
        maintainability = 'medium'
        refactoring_difficulty = 'medium'
    elif total_importers > 5:
        runtime_risk = 'low'
        testing_impact = 'medium'
        maintainability = 'medium'
        refactoring_difficulty = 'low'
    else:
        runtime_risk = 'low'
        testing_impact = 'low'
        maintainability = 'low'
        refactoring_difficulty = 'low'
    
    return {
        'runtime_risk': runtime_risk,
        'testing_impact': testing_impact,
        'maintainability': maintainability,
        'refactoring_difficulty': refactoring_difficulty
    }


def generate_circular_dependency_report(
    dependency_tree: Dict,
    import_data: Dict,
    cycles: List[List[str]]
) -> Dict:
    """Generate the complete circular dependency analysis report."""
    
    # Initialize counters
    total_cycles = len(cycles)
    direct_cycles = 0
    indirect_cycles = 0
    self_referential = 0
    critical_issues = 0
    high_issues = 0
    medium_issues = 0
    low_issues = 0
    
    modules_involved = set()
    modules_by_cycle_count = defaultdict(int)
    
    # Analyze each cycle
    circular_dependencies = []
    for idx, cycle in enumerate(cycles, 1):
        cycle_type = classify_cycle(cycle)
        severity = assess_severity(cycle, dependency_tree, import_data)
        impact = analyze_impact(cycle, dependency_tree)
        recommendation = generate_recommendation(cycle_type, severity, cycle)
        
        # Count by type
        if cycle_type == 'direct':
            direct_cycles += 1
        elif cycle_type == 'indirect':
            indirect_cycles += 1
        else:
            self_referential += 1
        
        # Count by severity
        if severity == 'critical':
            critical_issues += 1
        elif severity == 'high':
            high_issues += 1
        elif severity == 'medium':
            medium_issues += 1
        else:
            low_issues += 1
        
        # Track modules
        for module in cycle:
            modules_involved.add(module)
            modules_by_cycle_count[module] += 1
        
        # Build cycle modules with line numbers
        cycle_modules = []
        for i in range(len(cycle) - 1):
            from_module = cycle[i]
            to_module = cycle[i + 1]
            lines = get_import_line_numbers(import_data, from_module, to_module)
            
            cycle_modules.append({
                'file': from_module.replace('.', '/') + '.py',
                'line': lines[0] if lines else 0,
                'imports': to_module
            })
        
        # Add the last edge (back to start)
        from_module = cycle[-1]
        to_module = cycle[0]
        lines = get_import_line_numbers(import_data, from_module, to_module)
        cycle_modules.append({
            'file': from_module.replace('.', '/') + '.py',
            'line': lines[0] if lines else 0,
            'imports': to_module
        })
        
        circular_dependencies.append({
            'cycle_id': f'CD{idx:03d}',
            'type': cycle_type,
            'severity': severity,
            'modules': cycle_modules,
            'cycle_path': cycle,
            'cycle_length': len(cycle),
            'impact': impact,
            'recommendation': recommendation
        })
    
    return {
        'summary': {
            'total_circular_dependencies': total_cycles,
            'direct_cycles': direct_cycles,
            'indirect_cycles': indirect_cycles,
            'self_referential': self_referential,
            'modules_involved': sorted(list(modules_involved)),
            'critical_issues': critical_issues,
            'high_issues': high_issues,
            'medium_issues': medium_issues,
            'low_issues': low_issues
        },
        'circular_dependencies': circular_dependencies,
        'modules_by_cycle_count': dict(modules_by_cycle_count)
    }


def main():
    """Main execution function."""
    print("Loading dependency data...")
    
    # Load data
    dependency_tree = load_dependency_tree('plans/dependency_tree.json')
    import_data = load_import_data('plans/dependency_import_data.json')
    
    print(f"Loaded {len(dependency_tree.get('modules', {}))} modules from dependency tree")
    print(f"Loaded import data for {len(import_data.get('files', {}))} files")
    
    # Build import graph
    print("\nBuilding import graph...")
    graph = build_import_graph(import_data)
    print(f"Built graph with {len(graph)} nodes and {sum(len(v) for v in graph.values())} edges")
    
    # Detect cycles
    print("\nDetecting circular dependencies...")
    cycles = detect_cycles_dfs(graph)
    print(f"Found {len(cycles)} circular dependencies")
    
    # Generate report
    print("\nGenerating analysis report...")
    report = generate_circular_dependency_report(dependency_tree, import_data, cycles)
    
    # Save JSON report
    output_json = 'plans/circular_dependency_analysis.json'
    with open(output_json, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved JSON report to {output_json}")
    
    # Print summary
    print("\n" + "="*60)
    print("CIRCULAR DEPENDENCY ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total circular dependencies: {report['summary']['total_circular_dependencies']}")
    print(f"  - Direct cycles: {report['summary']['direct_cycles']}")
    print(f"  - Indirect cycles: {report['summary']['indirect_cycles']}")
    print(f"  - Self-referential: {report['summary']['self_referential']}")
    print("\nSeverity distribution:")
    print(f"  - Critical: {report['summary']['critical_issues']}")
    print(f"  - High: {report['summary']['high_issues']}")
    print(f"  - Medium: {report['summary']['medium_issues']}")
    print(f"  - Low: {report['summary']['low_issues']}")
    print(f"\nModules involved in cycles: {len(report['summary']['modules_involved'])}")
    
    if cycles:
        print("\n" + "="*60)
        print("CYCLES FOUND:")
        print("="*60)
        for cycle in cycles:
            print(f"  {' -> '.join(cycle)}")
    else:
        print("\n✓ No circular dependencies detected!")
    
    return report


if __name__ == '__main__':
    main()
