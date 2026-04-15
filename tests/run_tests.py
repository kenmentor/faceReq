#!/usr/bin/env python3
# tests/run_tests.py
"""
Test runner for the Face Recognition System.
Run all tests with: python run_tests.py
Run specific test file: python run_tests.py test_matching
Run with verbose: python run_tests.py -v
"""

import os
import sys
import unittest
import argparse


def setup_environment():
    """Set up the test environment."""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))


def discover_tests():
    """Discover all tests in the tests directory."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    test_files = [
        'test_matching.py',
        'test_detection.py', 
        'test_embedding.py',
        'test_database.py',
        'test_api.py'
    ]
    
    for test_file in test_files:
        test_path = os.path.join(test_dir, test_file)
        if os.path.exists(test_path):
            suite.addTests(loader.discover(
                os.path.dirname(test_path),
                pattern=test_file,
                top_level_dir=test_dir
            ))
    
    return suite


def run_unit_tests(verbose=False):
    """Run all unit tests."""
    setup_environment()
    
    print("=" * 60)
    print("Face Recognition System - Test Suite")
    print("=" * 60)
    print()
    
    suite = discover_tests()
    
    if verbose:
        runner = unittest.TextTestRunner(verbosity=2)
    else:
        runner = unittest.TextTestRunner(verbosity=1)
    
    result = runner.run(suite)
    
    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Success: {result.wasSuccessful()}")
    print("=" * 60)
    
    return 0 if result.wasSuccessful() else 1


def run_specific_test(test_name, verbose=False):
    """Run a specific test file."""
    setup_environment()
    
    loader = unittest.TestLoader()
    suite = loader.discover(
        os.path.dirname(os.path.abspath(__file__)),
        pattern=f'{test_name}.py',
        top_level_dir=os.path.dirname(os.path.abspath(__file__))
    )
    
    if verbose:
        runner = unittest.TextTestRunner(verbosity=2)
    else:
        runner = unittest.TextTestRunner(verbosity=1)
    
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


def run_with_pytest():
    """Run tests with pytest (if available)."""
    try:
        import pytest
    except ImportError:
        print("pytest not installed. Falling back to unittest.")
        return run_unit_tests()
    
    test_dir = os.path.dirname(os.path.abspath(__file__))
    exit_code = pytest.main([
        test_dir,
        '-v',
        '--tb=short',
        '--color=yes'
    ])
    return exit_code


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run Face Recognition System Tests')
    parser.add_argument('test', nargs='?', help='Specific test to run (without .py)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--pytest', action='store_true', help='Use pytest runner')
    
    args = parser.parse_args()
    
    if args.pytest:
        return run_with_pytest()
    elif args.test:
        return run_specific_test(args.test, args.verbose)
    else:
        return run_unit_tests(args.verbose)


if __name__ == "__main__":
    sys.exit(main())
