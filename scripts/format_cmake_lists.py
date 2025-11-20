#!/usr/bin/env python3
"""
Script to format GOMA_INCLUDES and GOMA_SOURCES in CMakeLists.txt alphabetically.
"""

import re
from pathlib import Path


def sort_cmake_list(content, list_name):
    """
    Sort a CMake list variable alphabetically.
    
    Args:
        content: Full CMakeLists.txt content
        list_name: Name of the list variable to sort (e.g., 'GOMA_INCLUDES')
    
    Returns:
        Updated content with sorted list
    """
    # Pattern to match the set(LIST_NAME ... ) block
    # Handles multi-line lists with proper indentation
    pattern = rf'set\({list_name}\s*\n(.*?)\)'
    
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print(f"Warning: {list_name} not found in CMakeLists.txt")
        return content
    
    # Extract the list content
    list_content = match.group(1)
    
    # Split into lines and filter out empty lines
    lines = [line.rstrip() for line in list_content.split('\n')]
    
    # Separate entries from empty lines and comments
    entries = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            entries.append(stripped)
    
    # Sort entries alphabetically (case-insensitive)
    entries.sort(key=str.lower)
    
    # Rebuild the list with proper indentation
    indent = '    '
    sorted_list = '\n'.join(f'{indent}{entry}' for entry in entries)
    
    # Replace the original list with the sorted one
    new_block = f'set({list_name}\n{sorted_list})'
    
    return content[:match.start()] + new_block + content[match.end():]


def main():
    cmake_file = Path(__file__).parent.parent / 'CMakeLists.txt'
    
    if not cmake_file.exists():
        print(f"Error: {cmake_file} not found")
        return 1
    
    print(f"Reading {cmake_file}")
    content = cmake_file.read_text()
    
    print("Sorting GOMA_INCLUDES...")
    content = sort_cmake_list(content, 'GOMA_INCLUDES')
    
    print("Sorting GOMA_SOURCES...")
    content = sort_cmake_list(content, 'GOMA_SOURCES')
    
    print(f"Writing sorted lists to {cmake_file}")
    cmake_file.write_text(content)
    
    print("Done! CMakeLists.txt has been updated.")
    return 0


if __name__ == '__main__':
    exit(main())
