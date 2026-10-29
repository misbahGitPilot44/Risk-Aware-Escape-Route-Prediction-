"""Launch the Pygame evacuation simulation."""

from visualization import run_app


def main() -> int:
    try:
        run_app()
    except RuntimeError as error:
        print(f"Cannot launch the graphical simulation: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
