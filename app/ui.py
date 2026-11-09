
"""Main graphical interface for AI Image Recognition Studio."""

import threading
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image

from app.classifier import ImageClassifier, Prediction
from app.config import (
    APP_TITLE,
    APPEARANCE_MODE,
    COLOR_THEME,
    PREVIEW_SIZE,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from app.camera import CameraManager
from app.config import CAMERA_UPDATE_DELAY
from PIL import ImageDraw
from app.image_processing import get_focus_box, crop_focus_area

class ImageRecognitionApp(ctk.CTk):
    """Manage the application interface and user interactions."""

    def __init__(self) -> None:
        super().__init__()

        self.classifier = None
        self.is_analyzing = False
        self.preview_image = None

        self.camera = CameraManager()
        self.camera_running = False
        self.current_frame = None
        self.camera_job = None

        self._configure_window()
        self._create_layout()
        self._load_classifier()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_window(self) -> None:
        """Configure the main application window."""
        ctk.set_appearance_mode(APPEARANCE_MODE)
        ctk.set_default_color_theme(COLOR_THEME)

        self.title(APP_TITLE)
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.minsize(900, 600)

    def _create_layout(self) -> None:
        """Create the main interface sections."""
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._create_sidebar()
        self._create_preview_panel()
        self._create_results_panel()

    def _create_sidebar(self) -> None:
        """Create navigation and action buttons."""
        sidebar = ctk.CTkFrame(self, width=220)
        sidebar.grid(
            row=0, column=0, sticky="nsew", padx=10, pady=10
        )
        sidebar.grid_propagate(False)

        title = ctk.CTkLabel(
            sidebar,
            text="AI Recognition",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        title.pack(pady=(30, 8))

        subtitle = ctk.CTkLabel(
            sidebar,
            text="Image Classification Studio",
            text_color="gray",
        )
        subtitle.pack(pady=(0, 30))

        
        # Upload image button
        self.upload_button = ctk.CTkButton(
            sidebar,
            text="Upload Image",
            command=self._upload_image,
            state="disabled",
        )
        self.upload_button.pack(
            fill="x", padx=20, pady=10
        )

        # Open camera button
        self.camera_button = ctk.CTkButton(
            sidebar,
            text="Open Camera",
            command=self._open_camera,
        )
        self.camera_button.pack(
            fill="x", padx=20, pady=10
        )

        # Capture photo button
        self.capture_button = ctk.CTkButton(
            sidebar,
            text="Capture Photo",
            command=self._capture_photo,
            state="disabled",
        )
        self.capture_button.pack(
            fill="x", padx=20, pady=10
        )

        # Close camera button
        self.close_camera_button = ctk.CTkButton(
            sidebar,
            text="Close Camera",
            command=self._close_camera,
            state="disabled",
        )
        self.close_camera_button.pack(
            fill="x", padx=20, pady=10
        )


        self.status_label = ctk.CTkLabel(
            sidebar,
            text="Loading AI model...",
            text_color="gray",
            wraplength=180,
        )
        self.status_label.pack(
            side="bottom", pady=25, padx=10
        )

    def _create_preview_panel(self) -> None:
        """Create the image preview area."""
        preview_frame = ctk.CTkFrame(self)
        preview_frame.grid(
            row=0, column=1, sticky="nsew", padx=5, pady=10
        )

        preview_frame.grid_rowconfigure(1, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)

        heading = ctk.CTkLabel(
            preview_frame,
            text="Image Preview",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        heading.grid(row=0, column=0, pady=20)

        self.image_label = ctk.CTkLabel(
            preview_frame,
            text="Upload an image to get started",
        )
        self.image_label.grid(
            row=1, column=0, padx=15, pady=15
        )

    def _create_results_panel(self) -> None:
        """Create the classification results area."""
        results_frame = ctk.CTkFrame(self, width=260)
        results_frame.grid(
            row=0, column=2, sticky="nsew", padx=10, pady=10
        )
        results_frame.grid_propagate(False)

        heading = ctk.CTkLabel(
            results_frame,
            text="AI Predictions",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        heading.pack(pady=25)

        self.results_label = ctk.CTkLabel(
            results_frame,
            text="No predictions yet",
            justify="left",
            anchor="w",
            wraplength=220,
        )
        self.results_label.pack(
            fill="x", padx=20, pady=10
        )

    def _load_classifier(self) -> None:
        """Load the model without blocking the interface."""
        threading.Thread(
            target=self._initialize_classifier,
            daemon=True,
        ).start()

    def _initialize_classifier(self) -> None:
        """Initialize the image classifier in a worker thread."""
        try:
            classifier = ImageClassifier()
        except Exception as error:
            self.after(
                0,
                self._show_model_error,
                str(error),
            )
            return

        self.after(0, self._on_model_ready, classifier)

    def _on_model_ready(
        self, classifier: ImageClassifier
    ) -> None:
        """Enable image upload after model initialization."""
        self.classifier = classifier
        self.upload_button.configure(state="normal")
        self.status_label.configure(text="Model ready")

    def _show_model_error(self, error: str) -> None:
        """Display a model loading error."""
        self.status_label.configure(text="Model loading failed")
        messagebox.showerror("Model Error", error)

    def _upload_image(self) -> None:
        """Open an image and start classification."""
        if self.is_analyzing:
            return

        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
            ]
        )

        if not file_path:
            return

        try:
            with Image.open(file_path) as source:
                image = source.convert("RGB")
        except (OSError, ValueError) as error:
            messagebox.showerror("Image Error", str(error))
            return

        self._display_image(image)
        self._start_analysis(image)

    def _display_image(self, image: Image.Image) -> None:
        """Display an image in the preview panel."""
        preview = image.copy()
        preview.thumbnail(PREVIEW_SIZE)

        self.preview_image = ctk.CTkImage(
            light_image=preview,
            dark_image=preview,
            size=preview.size,
        )

        self.image_label.configure(
            image=self.preview_image,
            text="",
        )

    def _start_analysis(self, image: Image.Image) -> None:
        """Start image classification in a background thread."""
        if self.classifier is None or self.is_analyzing:
            return

        self.is_analyzing = True
        self.upload_button.configure(state="disabled")
        self.status_label.configure(text="Analyzing image...")
        self.results_label.configure(text="Processing...")

        threading.Thread(
            target=self._analyze_image,
            args=(image,),
            daemon=True,
        ).start()

    def _analyze_image(self, image: Image.Image) -> None:
        """Run classification outside the interface thread."""
        try:
            predictions = self.classifier.classify_image(image)
        except Exception as error:
            self.after(
                0,
                self._show_analysis_error,
                str(error),
            )
            return

        self.after(
            0,
            self._display_predictions,
            predictions,
        )

    def _display_predictions(
        self, predictions: list[Prediction]
    ) -> None:
        """Display classification results."""
        result_lines = []

        for prediction in predictions:
            result_lines.append(
                f"{prediction.label}\n"
                f"Confidence: {prediction.confidence:.2f}%"
            )

        self.results_label.configure(
            text="\n\n".join(result_lines)
        )

        self.is_analyzing = False
        self.upload_button.configure(state="normal")
        self.status_label.configure(text="Analysis completed")

    def _show_analysis_error(self, error: str) -> None:
        """Handle classification errors."""
        self.is_analyzing = False
        self.upload_button.configure(state="normal")
        self.status_label.configure(text="Analysis failed")
        messagebox.showerror("Analysis Error", error)

    def _open_camera(self) -> None:
        """Open the camera and start the live preview."""
        if self.camera_running:
            return

        if not self.camera.open_camera():
            messagebox.showerror(
                "Camera Error",
                "Could not open the camera.",
            )
            return

        self.camera_running = True
        self.camera_button.configure(state="disabled")
        self.capture_button.configure(state="normal")
        self.close_camera_button.configure(state="normal")
        self.upload_button.configure(state="disabled")
        self.status_label.configure(text="Camera is running")

        self._update_camera_frame()

    def _update_camera_frame(self) -> None:
        """Read and display the next camera frame."""
        if not self.camera_running:
            return

        frame = self.camera.read_frame()

        if frame is None:
            self._close_camera()
            messagebox.showerror(
                "Camera Error",
                "Could not read a camera frame.",
            )
            return

        self.current_frame = Image.fromarray(frame)

        preview = self.current_frame.copy()
        draw = ImageDraw.Draw(preview)

        focus_box = get_focus_box(*preview.size)

        draw.rectangle(
     focus_box,
     outline="#00FF88",
     width=3,
          )

        self._display_image(preview) 

        self.camera_job = self.after(
            CAMERA_UPDATE_DELAY,
            self._update_camera_frame,
        )

    def _capture_photo(self) -> None:
        """Capture the current frame and classify it."""
        if self.current_frame is None or self.is_analyzing:
            return

        captured_image = crop_focus_area(self.current_frame)
        self._close_camera()
        self._display_image(captured_image)
        self._start_analysis(captured_image)

    def _close_camera(self) -> None:
        """Stop the preview and release the camera."""
        self.camera_running = False

        if self.camera_job is not None:
            self.after_cancel(self.camera_job)
            self.camera_job = None

        self.camera.close_camera()
        self.current_frame = None

        self.camera_button.configure(state="normal")
        self.capture_button.configure(state="disabled")
        self.close_camera_button.configure(state="disabled")

        if not self.is_analyzing and self.classifier is not None:
            self.upload_button.configure(state="normal")

        self.status_label.configure(text="Camera closed")
    def _on_close(self) -> None:
     """Release camera resources before exiting."""
     if self.camera_running:
        self._close_camera()

     self.destroy()    
