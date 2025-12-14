"""
argument_parser.py

A small argument-parsing utility module.
"""

from __future__ import annotations

import argparse
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    """
    Create and configure an ArgumentParser for the application.

    Returns:
        argparse.ArgumentParser: The configured parser instance.
    """
    parser = argparse.ArgumentParser(
        description="Example argument parser showing typed arguments."
    )

    parser.add_argument(
        "--user-input",
        type=str,
        default="Harry",
        help="Enter the name for display.",
    )

    parser.add_argument(
        "--age",
        type=int,
        default=20,    # default should match type, so integer not string
        help="Enter numeric age for display.",
    )

    return parser


def parse_arguments(args: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments using the configured parser.

    Args:
        args (list[str] | None): Optional list of arguments for testing.
                                 If None, argparse parses sys.argv.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = build_parser()
    return parser.parse_args(args)


def main() -> None:
    """Entry point for manual execution."""
    args = parse_arguments()
    print(f"The entered name is {args.user_input} of age {args.age}")


if __name__ == "__main__":
    main()
