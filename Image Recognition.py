"""Entry point for the AI Image Recognition application."""

from app.ui import ImageRecognitionApp


def main() -> None:
    """Start the application."""
    app = ImageRecognitionApp()
    app.mainloop()


if __name__ == "__main__":
    main()