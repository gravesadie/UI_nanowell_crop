import os
import sys
import glob
from pathlib import Path

# Suppress low-level OpenCV C++ warnings and libtiff logs before importing cv2
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QTextEdit, QFileDialog, QGridLayout, QHBoxLayout, QVBoxLayout, QSplitter,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QCheckBox, QStackedWidget, 
    QRadioButton, QButtonGroup, QProgressBar, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QWheelEvent, QPainter
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

import core_crop
import image_analysis
from mcherry_analysis import (
    load_mcherry_crop_and_mask,
    analyze_mcherry_image,
    plot_mcherry_analysis,
    batch_analyze_mcherry,
    save_mcherry_figure
)


import torch


class InteractiveView(QGraphicsView):
    """
    Interactive canvas supporting mouse wheel zooming centered on cursor
    and scroll-hand drag panning for alignment verification.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = None

        # Configure the interactive drag mode to scroll-hand dragging
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    
    def set_image(self, pixmap: QPixmap):
        # Updates the background image while maintaining the current zoom state.
        self.scene.clear()
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        # Enable bilinear filtering to keep edges smooth when zooming in on nanowells
        self.pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene.addItem(self.pixmap_item)
        self.scene.setSceneRect(self.scene.itemsBoundingRect())

    def wheelEvent(self, event: QWheelEvent):
        # Intercepts the mouse wheel event to zoom centered on the cursor position.
        if self.pixmap_item is None:
            return
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        scale_factor = 1.15 if event.angleDelta().y() > 0 else (1.0 / 1.15)
        self.scale(scale_factor, scale_factor)


# Dedicated worker thread for Cellpose AI inference to prevent UI freezing
class AIWorkerThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)  # (current, total)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, proc_dir, well_name, time, mode, model_dir, save_masks, cell_diameter):
        super().__init__()
        self.proc_dir = proc_dir
        self.well_name = well_name
        self.time = time
        self.mode = mode
        self.model_dir = model_dir
        self.save_masks = save_masks
        self.cell_diameter = cell_diameter

    def run(self):
        try:
            image_analysis.execute_ai_segmentation(
                processed_wells_dir=self.proc_dir,
                well_name=self.well_name,
                time=self.time,
                mode=self.mode,
                model_dir=self.model_dir,
                save_masks=self.save_masks,
                cell_diameter = self.cell_diameter,
                log_callback=self.log_signal.emit,
                progress_callback=self.progress_signal.emit
            )
        except Exception as e:
            self.error_signal.emit(str(e))
        finally:
            self.finished_signal.emit()


# ==============================================================================
# Main GUI Window
# ==============================================================================

class MicroscopyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Microscopy Nanowell Processor & Crop Tool")
        self.resize(1400, 900)

        # In-memory image caching structures
        self.cached_gray = None
        self.cached_gray_crop = None
        self.cached_bgr = None
        self.cached_path = ""
        self.valid_wells = []

        # Local model directory path (auto-adapts to frozen .exe or source)
        if getattr(sys, 'frozen', False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(base_dir, "models")

        # Worker thread holder
        self.ai_thread = None

        self.init_ui()
        self.log("System initialized. Ready for operations.")

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # ---------------- Left Panel: Multi-page Stacked Widget ----------------
        self.left_stack = QStackedWidget()

        page1_widget = self.build_page1_cropping_ui()
        page2_widget = self.build_page2_analysis_ui()
        page3_widget = self.build_page3_mcherry_ui()

        self.left_stack.addWidget(page1_widget)  # Index 0: Cropping Page
        self.left_stack.addWidget(page2_widget)  # Index 1: AI segmentation Page
        self.left_stack.addWidget(page3_widget)  # Index 2: mCherry puncta analysis page

        splitter.addWidget(self.left_stack)

        # ---------------- Right Canvas & Console Panel ----------------
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        right_layout.addWidget(QLabel("<b>Real-time Interactive Mask Preview</b>"))
        self.canvas = InteractiveView(self)
        self.canvas.setMinimumSize(600, 500)
        self.canvas.setStyleSheet("background-color: #1B2631; border: 2px dashed #34495E;")
        right_layout.addWidget(self.canvas, stretch=4)

        right_layout.addWidget(QLabel("<b>Operation System Console Log</b>"))
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet(
            "background-color: #FFFFFF;"
            "color: #1A252C;"
            "font-family: Consolas, 'Courier New', monospace;"
            "font-size: 12px;"
            "border: 1px solid #BDC3C7;"
            "border-radius: 4px;"
            "padding: 6px;"
        )
        right_layout.addWidget(self.console, stretch=1)

        splitter.addWidget(right_panel)
        splitter.setSizes([450, 950])

    # Page 1: Left Control Panel 
    def build_page1_cropping_ui(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Step 1: Rename Section
        layout.addWidget(QLabel("<b>Step 1: Filename Standardizer</b>"))
        self.rename_dir_input = QLineEdit()
        self.rename_dir_input.setPlaceholderText("Select the directory containing raw TIFs...")
        self.btn_browse_rename = QPushButton("Browse")
        self.btn_browse_rename.clicked.connect(lambda: self.browse_folder(self.rename_dir_input))

        h_rename1 = QHBoxLayout()
        h_rename1.addWidget(self.rename_dir_input)
        h_rename1.addWidget(self.btn_browse_rename)
        layout.addLayout(h_rename1)

        self.rename_time = QLineEdit()
        self.rename_time.setPlaceholderText("e.g., 1 or 2")
        self.btn_run_rename = QPushButton("Run Rename Task")
        self.btn_run_rename.setStyleSheet("background-color: #2E86C1; color: white; font-weight: bold;")
        self.btn_run_rename.clicked.connect(self.on_run_rename)

        h_rename2 = QHBoxLayout()
        h_rename2.addWidget(QLabel("Time Index:"))
        h_rename2.addWidget(self.rename_time)
        h_rename2.addWidget(self.btn_run_rename)
        layout.addLayout(h_rename2)

        layout.addWidget(QLabel("<hr>"))

        # Step 2: Parameter Configuration Section
        layout.addWidget(QLabel("<b>Step 2: Parameter Configuration</b>"))
        grid = QGridLayout()

        grid.addWidget(QLabel("Well Name:"), 0, 0)
        self.in_well_name = QLineEdit()
        self.in_well_name.setPlaceholderText("e.g. A02 or C05")
        grid.addWidget(self.in_well_name, 0, 1)

        grid.addWidget(QLabel("Time Index:"), 1, 0)
        self.in_time = QLineEdit()
        self.in_time.setPlaceholderText("e.g., 1 or 2")
        grid.addWidget(self.in_time, 1, 1)

        grid.addWidget(QLabel("Image Folder:"), 2, 0)
        self.in_img_dir = QLineEdit()
        self.btn_browse_img = QPushButton("Browse")
        self.btn_browse_img.clicked.connect(lambda: self.browse_folder(self.in_img_dir))
        h_img = QHBoxLayout()
        h_img.addWidget(self.in_img_dir)
        h_img.addWidget(self.btn_browse_img)
        grid.addLayout(h_img, 2, 1)

        grid.addWidget(QLabel("Nanowell R (px):"), 3, 0)
        self.in_well_r = QLineEdit('200')
        self.in_well_r.setToolTip("Individual Well Radius (pixels)")
        grid.addWidget(self.in_well_r, 3, 1)

        grid.addWidget(QLabel("Boundary R (px):"), 4, 0)
        self.in_bound_r = QLineEdit('9000')
        self.in_bound_r.setToolTip("Boundary outer radius (pixels)")
        grid.addWidget(self.in_bound_r, 4, 1)

        grid.addWidget(QLabel("Square Length (px):"), 5, 0)
        self.in_sq_len = QLineEdit('370')
        self.in_sq_len.setToolTip("Origin marker side length (pixels)")
        grid.addWidget(self.in_sq_len, 5, 1)

        grid.addWidget(QLabel("Pitch (px):"), 6, 0)
        self.in_pitch = QLineEdit("462")
        self.in_pitch.setToolTip("Adjacent center-to-center well pitch (pixels)")
        grid.addWidget(self.in_pitch, 6, 1)
        layout.addLayout(grid)

        # Step 3: Calculation & Coordinates Section
        self.btn_calc = QPushButton("📊 Calculate Initial Parameters")
        self.btn_calc.setStyleSheet("background-color: #27AE60; color: white; font-weight: bold;")
        self.btn_calc.clicked.connect(self.on_run_calculation)
        layout.addWidget(self.btn_calc)

        grid_calc = QGridLayout()
        grid_calc.addWidget(QLabel("Calculated Angle (°):"), 0, 0)
        self.out_angle = QLineEdit()
        grid_calc.addWidget(self.out_angle, 0, 1)

        grid_calc.addWidget(QLabel("Center X (px):"), 1, 0)
        self.out_cx = QLineEdit()
        grid_calc.addWidget(self.out_cx, 1, 1)

        grid_calc.addWidget(QLabel("Center Y (px):"), 2, 0)
        self.out_cy = QLineEdit()
        grid_calc.addWidget(self.out_cy, 2, 1)
        layout.addLayout(grid_calc)

        # Step 4: Visualization and Cropping Section
        self.btn_visualize = QPushButton("👁️ Visualize / Update Preview")
        self.btn_visualize.setStyleSheet("background-color: #E67E22; color: white; font-weight: bold;")
        self.btn_visualize.clicked.connect(self.on_run_visualization)
        layout.addWidget(self.btn_visualize)

        crop_size_container = QWidget()
        h_crop_size = QHBoxLayout(crop_size_container)
        h_crop_size.setContentsMargins(0, 5, 0, 5)
        h_crop_size.addWidget(QLabel("Crop Resolution (px):"))

        self.cb_default_size = QCheckBox("400")
        self.cb_default_size.setChecked(True)
        self.cb_other_size = QCheckBox("Other:")
        self.in_custom_size = QLineEdit()
        self.in_custom_size.setPlaceholderText("Enter size...")
        self.in_custom_size.setEnabled(False)

        self.cb_default_size.toggled.connect(self.on_default_size_toggled)
        self.cb_other_size.toggled.connect(self.on_other_size_toggled)

        h_crop_size.addWidget(self.cb_default_size)
        h_crop_size.addWidget(self.cb_other_size)
        h_crop_size.addWidget(self.in_custom_size)
        layout.addWidget(crop_size_container)

        self.btn_crop = QPushButton("✂️ Crop & Export All Nanowells")
        self.btn_crop.setStyleSheet("background-color: #8E44AD; color: white; font-weight: bold; font-size: 14px;")
        self.btn_crop.clicked.connect(self.on_run_cropping)
        layout.addWidget(self.btn_crop)

        # Step 5: Rollback Engine Section
        layout.addWidget(QLabel("<br><hr style='border: 1px dashed #E74C3C;'>"))
        rollback_title = QLabel("<b>⚠️ Emergency Rollback & Purge Engine</b>")
        rollback_title.setStyleSheet("color: #E67E22;")
        layout.addWidget(rollback_title)

        rollback_desc = QLabel("Deletes all exported single-well crops matching current inputs.")
        rollback_desc.setStyleSheet("font-size: 10px; color: #2C3E50;")
        layout.addWidget(rollback_desc)

        self.btn_rollback = QPushButton("🗑️ Purge Matching Cropped Wells")
        self.btn_rollback.setStyleSheet("background-color: #7B241C; color: #F5EEF8; font-weight: bold; font-size: 11px; border: 1px solid #C0392B;")
        self.btn_rollback.clicked.connect(self.on_run_rollback)
        layout.addWidget(self.btn_rollback)

        # Push bottom navigation bar down
        layout.addStretch()

        # Bottom Page Navigation Button
        layout.addWidget(QLabel("<hr style='border: 0; border-top: 1px solid #BDC3C7;'>"))
        self.btn_next_page = QPushButton("Next: Image Analysis ➡️")
        self.btn_next_page.setStyleSheet("background-color: #117A65; color: white; font-weight: bold; font-size: 13px; padding: 8px; border-radius: 4px;")
        self.btn_next_page.clicked.connect(self.switch_to_page2)
        layout.addWidget(self.btn_next_page)

        return panel

    # Page 2: Left Control Panel 
    def build_page2_analysis_ui(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(QLabel("<b>🔬 Step 3: AI Cell Segmentation & Quantification</b>"))
        layout.addWidget(QLabel("<span style='font-size:10px; color:#5D6D7E;'>Powered by Cellpose cpsam_v2 Deep Learning Model</span>"))

        grid_ai = QGridLayout()
        grid_ai.addWidget(QLabel("Processed Wells Dir:"), 0, 0)
        self.ai_processed_dir = QLineEdit()
        self.ai_processed_dir.setPlaceholderText("Path to 'Processed Wells' directory...")
        self.btn_browse_ai_dir = QPushButton("Browse")
        self.btn_browse_ai_dir.clicked.connect(lambda: self.browse_folder(self.ai_processed_dir))
        h_dir = QHBoxLayout()
        h_dir.addWidget(self.ai_processed_dir)
        h_dir.addWidget(self.btn_browse_ai_dir)
        grid_ai.addLayout(h_dir, 0, 1)

        grid_ai.addWidget(QLabel("Well Name:"), 1, 0)
        self.ai_well_name = QLineEdit()
        self.ai_well_name.setPlaceholderText("e.g. A02")
        grid_ai.addWidget(self.ai_well_name, 1, 1)

        grid_ai.addWidget(QLabel("Time Index:"), 2, 0)
        self.ai_time = QLineEdit()
        self.ai_time.setPlaceholderText("e.g., 1 or 2")
        grid_ai.addWidget(self.ai_time, 2, 1)

        layout.addLayout(grid_ai)
        layout.addWidget(QLabel("<hr>"))

        # Mode Selection
        layout.addWidget(QLabel("<b>Cell Counting & Segmentation Target:</b>"))
        self.mode_group = QButtonGroup(self)

        self.rb_mode1 = QRadioButton("Mode 1: Find & process single-cell nanowells only (Count == 1)")
        self.rb_mode2 = QRadioButton("Mode 2: Process designated coordinates from Excel file")
        self.rb_mode3 = QRadioButton("Mode 3: Process all cropped nanowells")
        self.rb_mode3.setChecked(True)   

        self.mode_group.addButton(self.rb_mode1, 1)
        self.mode_group.addButton(self.rb_mode2, 2)
        self.mode_group.addButton(self.rb_mode3, 3)

        layout.addWidget(self.rb_mode1)
        layout.addWidget(self.rb_mode2)
        layout.addWidget(self.rb_mode3)

        layout.addSpacing(6)

        # ---------- Cell Diameter Selection -----
        layout.addWidget(QLabel("<b>Cell Diameter (px):</b>"))
        diam_container = QWidget()
        h_diam = QHBoxLayout(diam_container)
        h_diam.setContentsMargins(0, 0, 0, 0)
        h_diam.setSpacing(10)

        self.cb_diam_default = QCheckBox("30 px (Default)")
        self.cb_diam_default.setChecked(True)

        self.cb_diam_other = QCheckBox("Other:")
        self.in_custom_diam = QLineEdit()
        self.in_custom_diam.setPlaceholderText("e.g. 32, 59")
        self.in_custom_diam.setEnabled(False)
        self.in_custom_diam.setMaximumWidth(90)

        # Binding mutually exclusive events
        self.cb_diam_default.toggled.connect(self.on_diam_default_toggled)
        self.cb_diam_other.toggled.connect(self.on_diam_other_toggled)

        h_diam.addWidget(self.cb_diam_default)
        h_diam.addWidget(self.cb_diam_other)
        h_diam.addWidget(self.in_custom_diam)
        h_diam.addStretch()
        layout.addWidget(diam_container)

        layout.addSpacing(6)

        # -------- Export Mask CheckBox ------------
        self.cb_save_masks = QCheckBox("Save validation mask images (Original & Colored Mask pairs)")
        self.cb_save_masks.setStyleSheet("font-weight: bold; color: #2C3E50;")
        layout.addWidget(self.cb_save_masks)

        layout.addSpacing(6)

        # AI analysis progress bar
        self.ai_progress_bar = QProgressBar()
        self.ai_progress_bar.setRange(0, 100)
        self.ai_progress_bar.setValue(0)
        self.ai_progress_bar.setTextVisible(True)
        self.ai_progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #BDC3C7;
                border-radius: 4px;
                text-align: center;
                height: 18px;
                background-color: #ECF0F1;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #27AE60;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.ai_progress_bar)

        # Run AI Analysis Button
        self.btn_run_ai = QPushButton("🧠 Run AI Segmentation & Update Excel")
        self.btn_run_ai.setStyleSheet("background-color: #8E44AD; color: white; font-weight: bold; font-size: 14px; padding: 10px; border-radius: 4px;")
        self.btn_run_ai.clicked.connect(self.on_run_ai_analysis)
        layout.addWidget(self.btn_run_ai)

        # Push bottom navigation bar down
        layout.addStretch()

        # Bottom Page Back Navigation Button
        layout.addWidget(QLabel("<hr style='border: 0; border-top: 1px solid #BDC3C7;'>"))
        self.btn_back_page = QPushButton("⬅️ Back: Nanowell Cropping")
        self.btn_back_page.setStyleSheet("background-color: #5D6D7E; color: white; font-weight: bold; font-size: 12px; padding: 8px; border-radius: 4px;")
        self.btn_back_page.clicked.connect(self.switch_to_page1)
        layout.addWidget(self.btn_back_page)

        # navigate to mCherry analysis
        self.btn_mcherry_page = QPushButton(
            "🔴 Continue to mCherry Puncta Analysis"
        )
        self.btn_mcherry_page.setStyleSheet(
            "background-color: #C0392B; "
            "color: white; "
            "font-weight: bold; "
            "font-size: 13px; "
            "padding: 9px; "
            "border-radius: 4px;"
        )
        self.btn_mcherry_page.clicked.connect(
            self.switch_to_page3
        )
        layout.addWidget(self.btn_mcherry_page)

        return panel

    def build_page3_mcherry_ui(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(QLabel("<b>🔴 Step 4: mCherry Puncta Analysis</b>"))

        grid = QGridLayout()

        grid.addWidget(QLabel("mCherry Crops Dir:"), 0, 0)
        self.mcherry_crop_dir = QLineEdit()
        self.btn_browse_mcherry_crop = QPushButton("Browse")
        self.btn_browse_mcherry_crop.clicked.connect(
            lambda: self.browse_folder(self.mcherry_crop_dir)
        )
        h = QHBoxLayout()
        h.addWidget(self.mcherry_crop_dir)
        h.addWidget(self.btn_browse_mcherry_crop)
        grid.addLayout(h, 0, 1)

        grid.addWidget(QLabel("Cell Masks Dir:"), 1, 0)
        self.mcherry_mask_dir = QLineEdit()
        self.btn_browse_mcherry_mask = QPushButton("Browse")
        self.btn_browse_mcherry_mask.clicked.connect(
            lambda: self.browse_folder(self.mcherry_mask_dir)
        )
        h = QHBoxLayout()
        h.addWidget(self.mcherry_mask_dir)
        h.addWidget(self.btn_browse_mcherry_mask)
        grid.addLayout(h, 1, 1)

        grid.addWidget(QLabel("Output Dir:"), 2, 0)
        self.mcherry_output_dir = QLineEdit()
        self.btn_browse_mcherry_output = QPushButton("Browse")
        self.btn_browse_mcherry_output.clicked.connect(
            lambda: self.browse_folder(self.mcherry_output_dir)
        )
        h = QHBoxLayout()
        h.addWidget(self.mcherry_output_dir)
        h.addWidget(self.btn_browse_mcherry_output)
        grid.addLayout(h, 2, 1)

        layout.addLayout(grid)

        params = QGridLayout()

        params.addWidget(QLabel("Min Diameter:"), 0, 0)
        self.mcherry_min_diameter = QLineEdit("2.6")
        params.addWidget(self.mcherry_min_diameter, 0, 1)

        params.addWidget(QLabel("Max Diameter:"), 0, 2)
        self.mcherry_max_diameter = QLineEdit("8")
        params.addWidget(self.mcherry_max_diameter, 0, 3)

        params.addWidget(QLabel("Threshold:"), 1, 0)
        self.mcherry_threshold = QLineEdit("0.09")
        params.addWidget(self.mcherry_threshold, 1, 1)

        self.btn_mcherry_start = QPushButton("🟢 Start")
        self.btn_mcherry_start.clicked.connect(
            self.initialize_mcherry_analysis
        )
        params.addWidget(self.btn_mcherry_start, 1, 2)

        self.btn_mcherry_replot = QPushButton("🔄 Replot")
        self.btn_mcherry_replot.clicked.connect(
            self.replot_current_mcherry
        )
        params.addWidget(self.btn_mcherry_replot, 1, 3)

        layout.addLayout(params)

        self.mcherry_current_label = QLabel("No crop loaded")
        self.mcherry_count_label = QLabel("Puncta: --")

        info = QHBoxLayout()
        info.addWidget(self.mcherry_current_label)
        info.addStretch()
        info.addWidget(self.mcherry_count_label)
        layout.addLayout(info)

        self.mcherry_figure = Figure(figsize=(7, 6))
        self.mcherry_canvas = FigureCanvas(self.mcherry_figure)
        layout.addWidget(self.mcherry_canvas, 1)

        nav = QHBoxLayout()

        self.btn_mcherry_previous = QPushButton("⬅ Previous")
        self.btn_mcherry_previous.clicked.connect(
            self.previous_mcherry_crop
        )

        self.btn_mcherry_next = QPushButton("Next ➡")
        self.btn_mcherry_next.clicked.connect(
            self.next_mcherry_crop
        )

        self.btn_mcherry_save = QPushButton("💾 Save Visualization")
        self.btn_mcherry_save.clicked.connect(
            self.save_current_mcherry_visualization
        )

        nav.addWidget(self.btn_mcherry_previous)
        nav.addWidget(self.btn_mcherry_next)
        nav.addWidget(self.btn_mcherry_save)

        layout.addLayout(nav)

        self.mcherry_progress_bar = QProgressBar()
        layout.addWidget(self.mcherry_progress_bar)

        self.btn_mcherry_batch = QPushButton(
            "🔬 Run Batch mCherry Analysis"
        )
        self.btn_mcherry_batch.clicked.connect(
            self.run_batch_mcherry_analysis
        )
        layout.addWidget(self.btn_mcherry_batch)

        self.btn_back_mcherry = QPushButton(
            "⬅️ Back: AI Cell Segmentation"
        )
        self.btn_back_mcherry.clicked.connect(
            self.switch_to_page2
        )
        layout.addWidget(self.btn_back_mcherry)

        self.mcherry_files = []
        self.mcherry_current_index = -1
        self.mcherry_current_result = None

        return panel

    def switch_to_page2(self):
        """Pre-fills Page 2 inputs from Page 1, dynamically detects Excel presence, and switches page."""
        load_path = self.in_img_dir.text().strip().replace('\\', '/')
        if load_path:
            inferred_proc_dir = os.path.join(Path(load_path).parent, "Processed Wells")
            self.ai_processed_dir.setText(inferred_proc_dir)

        self.ai_well_name.setText(self.in_well_name.text().strip())
        self.ai_time.setText(self.in_time.text().strip())

        # Select Mode 2 by default if historical Excel database exists; otherwise Mode 1
        proc_dir = self.ai_processed_dir.text().strip().replace('\\', '/')
        well_name = self.ai_well_name.text().strip()
        if proc_dir and well_name:
            excel_path = os.path.join(proc_dir, f"{well_name}_CellCount.xlsx")
            if os.path.exists(excel_path):
                self.rb_mode2.setChecked(True)
                self.log(f"[INFO]: Found existing '{well_name}_CellCount.xlsx'. Mode 2 selected by default.")
            else:
                self.rb_mode1.setChecked(True)
        else:
            self.rb_mode1.setChecked(True)

        self.left_stack.setCurrentIndex(1)
        self.log("[NAV]: Switched to AI Image Analysis Workspace.")

    def switch_to_page1(self):
        self.left_stack.setCurrentIndex(0)
        self.log("[NAV]: Switched back to Nanowell Cropping Workspace.")

    def switch_to_page3(self):
        self.left_stack.setCurrentIndex(2)

        crop_dir = self.mcherry_crop_dir.text().strip()

        if crop_dir and os.path.isdir(crop_dir):
            self.initialize_mcherry_analysis()

        self.log("[NAV]: Switched to mCherry Analysis Workspace.")

    def log(self, text: str):
        """Logs message and immediately forces Qt event loop to update GUI widgets."""
        self.console.append(text)
        self.console.moveCursor(self.console.textCursor().MoveOperation.End)
        QApplication.processEvents()

    def browse_folder(self, target_lineedit: QLineEdit):
        # 1. try to start with the path in the textbox
        current_text = target_lineedit.text().strip().replace('\\', '/')
        start_dir = ""

        if current_text and os.path.exists(current_text):
            if os.path.isdir(current_text):
                start_dir = os.path.dirname(current_text) or current_text
            else:
                start_dir = os.path.dirname(current_text)

        # 2. Empty textbox => go to the .exe installation directory
        if not start_dir or not os.path.exists(start_dir):
            if getattr(sys, 'frozen', False):
                start_dir = os.path.dirname(sys.executable)
            else:
                start_dir = os.path.dirname(os.path.abspath(__file__))

        folder = QFileDialog.getExistingDirectory(self, "Select Directory", start_dir)
        if folder:
            target_lineedit.setText(folder.replace('\\', '/'))

    def ensure_image_loaded(self) -> bool:
        img_dir = self.in_img_dir.text().strip()
        well_name = self.in_well_name.text().strip()
        time = self.in_time.text().strip()

        if not img_dir or not well_name or not time:
            self.log("❌ [WARNING]: Image Folder, Well Name, and Time Index must all be filled!")
            return False

        bgr, gray, full_path, err = core_crop.load_bf_image(img_dir, well_name, time)
        if err:
            self.log(f"❌ [ERROR]: {err}")
            return False

        if self.cached_path != full_path:
            self.cached_bgr = bgr
            self.cached_gray = gray
            self.cached_path = full_path
            self.cached_gray_crop = None
            self.log(f"✅ [SUCCESS]: Resolution loaded: {gray.shape[1]}x{gray.shape[0]}")

        return True

    def on_default_size_toggled(self, checked: bool):
        if checked:
            self.cb_other_size.blockSignals(True)
            self.cb_other_size.setChecked(False)
            self.cb_other_size.blockSignals(False)
            self.in_custom_size.setEnabled(False)
            self.in_custom_size.clear()

    def on_other_size_toggled(self, checked: bool):
        if checked:
            self.cb_default_size.blockSignals(True)
            self.cb_default_size.setChecked(False)
            self.cb_default_size.blockSignals(False)
            self.in_custom_size.setEnabled(True)
            self.log("[ATTENTION]: Custom dimension selected. Verify consistency for downstream pipelines.")
        elif not self.cb_default_size.isChecked():
            self.cb_default_size.setChecked(True)

    def on_run_rename(self):
        raw_dir = self.rename_dir_input.text().strip()
        time = self.rename_time.text().strip()
        if not raw_dir or not time:
            self.log("⚠️ [WARNING]: Target raw directory and Time index are required for renaming!")
            return

        self.btn_run_rename.setEnabled(False)
        self.log(f"[STANDARDIZE]: Renaming image files in '{raw_dir}' for Time {time}...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            core_crop.rename_raw_files(
                directory=raw_dir,
                time=time,
                log_callback=self.log
            )
        finally:
            QApplication.restoreOverrideCursor()
            self.btn_run_rename.setEnabled(True)

    def on_run_calculation(self):
        self.btn_calc.setEnabled(False)
        self.log("[CALCULATING]: Detecting origin square and estimating array rotation angle...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            if not self.ensure_image_loaded():
                return

            try:
                sq_len = int(self.in_sq_len.text().strip())
                nanowell_r = int(self.in_well_r.text().strip())
            except ValueError:
                self.log("❌ [WARNING]: Nanowell R and Square Length must be valid integers!")
                return

            # 1. Square origin detection
            cx_text = self.out_cx.text().strip()
            cy_text = self.out_cy.text().strip()

            if cx_text and cy_text:
                self.log(f"ℹ️ [MANUAL]: Using user-defined Origin at X:{cx_text}, Y:{cy_text}")
            else:
                rect_center, self.cached_gray_crop, err = core_crop.detect_center_square(self.cached_gray, sq_len)
                if rect_center:
                    self.out_cx.setText(str(rect_center[0]))
                    self.out_cy.setText(str(rect_center[1]))
                    self.log(f"✅ [FOUND]: Central Origin calibrated at X:{rect_center[0]}, Y:{rect_center[1]}")
                else:
                    self.log(f"❌ [ERROR]: {err}")
                    return

            # 2. Grid rotation angle estimation
            angle_text = self.out_angle.text().strip()
            if angle_text:
                self.log(f"ℹ️ [MANUAL]: Using user-defined angle = {angle_text}°")
            else:
                angle, err = core_crop.detect_array_angle(self.cached_gray, nanowell_r)
                if angle is not None:
                    self.out_angle.setText(f"{angle:.4f}")
                    self.log(f"✅ [FOUND]: Grid rotation angle calculated: {angle:.4f}°")
                else:
                    self.log(f"❌ [ERROR]: {err}")
        finally:
            QApplication.restoreOverrideCursor()
            self.btn_calc.setEnabled(True)

    def on_run_visualization(self):
        self.btn_visualize.setEnabled(False)
        self.log("[VISUALIZING]: Regenerating math model grid overlay...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            if not self.ensure_image_loaded():
                return

            try:
                cx = int(self.out_cx.text().strip())
                cy = int(self.out_cy.text().strip())
                angle_val = float(self.out_angle.text().strip())
                bound_r = float(self.in_bound_r.text().strip())
                pitch = float(self.in_pitch.text().strip())
                nanowell_r = int(self.in_well_r.text().strip())
            except ValueError:
                self.log("❌ [ERROR]: Parsing failed. Check Angle, Pitch, Boundary R, Nanowell R, and Coordinates.")
                return

            self.valid_wells, render_img = core_crop.generate_grid_overlay(
                self.cached_bgr, cx, cy, angle_val, bound_r, pitch, nanowell_r
            )

            h, w, ch = render_img.shape
            bytes_per_line = ch * w
            qimg = QImage(render_img.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
            self.canvas.set_image(QPixmap.fromImage(qimg))
            self.log(f"✅ [SUCCESS]: Preview updated. Array micro-nodes tracked: {len(self.valid_wells)}")
        finally:
            QApplication.restoreOverrideCursor()
            self.btn_visualize.setEnabled(True)

    def on_run_cropping(self):
        if not self.valid_wells:
            self.log("❌ [ERROR]: Layout grid is empty. Click 'Visualize' beforehand.")
            return

        try:
            nanowell_r = int(self.in_well_r.text().strip())
            if self.cb_other_size.isChecked():
                output_size = int(self.in_custom_size.text().strip())
            else:
                output_size = 400
        except ValueError:
            self.log("❌ [ERROR]: Invalid pixel dimensions specified.")
            return

        if output_size < (2 * nanowell_r):
            self.log(f"❌ [CRITICAL ABORT]: Output size ({output_size}px) is smaller than diameter ({2 * nanowell_r}px).")
            return

        well_name = self.in_well_name.text().strip()
        time = self.in_time.text().strip()
        load_path = self.in_img_dir.text().strip().replace('\\', '/')
        output_dir = os.path.join(os.path.dirname(load_path), "Processed Wells")
        well_target_dir = os.path.join(output_dir, well_name)

        # 1. check if the cropped image already exists
        existing_crops = glob.glob(os.path.join(well_target_dir, "*", f"{well_name}_*_Time{time}_*.png"))
        
        if existing_crops:
            reply = QMessageBox.question(
                self,
                "Clean & Overwrite Existing Crops",
                f"Found {len(existing_crops)} existing crop files for Well '{well_name}' (Time {time}).\n\n"
                f"Do you want to purge all old crops and export the newly aligned {len(self.valid_wells)} nanowells?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No  # highlight "No"
            )

            if reply == QMessageBox.StandardButton.No:
                self.log("ℹ️ [CANCELLED]: Re-cropping operation cancelled by user.")
                return

            # If "YES", remove all old images and save newly cropped images
            self.log(f"[CLEANUP]: Purging previous crops for Time {time} before writing fresh crops...")
            core_crop.execute_rollback(load_path, well_name, time, log_callback=self.log)

        # start cropping
        self.btn_crop.setEnabled(False)
        self.btn_crop.setText("⏳ Cropping in Progress...")
        self.log("[CROPPING]: Starting batch multi-channel cropping...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            core_crop.execute_nanowell_crop(
                load_path=self.in_img_dir.text(),
                well_name=self.in_well_name.text(),
                time=self.in_time.text(),
                nanowell_r=nanowell_r,
                output_size=output_size,
                valid_wells=self.valid_wells,
                log_callback=self.log
            )
        finally:
            QApplication.restoreOverrideCursor()
            self.btn_crop.setEnabled(True)
            self.btn_crop.setText("✂️ Crop & Export All Nanowells")

    def on_run_rollback(self):
        self.btn_rollback.setEnabled(False)
        self.log(f"[ROLLBACK]: Purging matching cropped images...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            core_crop.execute_rollback(
                load_path=self.in_img_dir.text(),
                well_name=self.in_well_name.text(),
                time=self.in_time.text(),
                log_callback=self.log
            )
        finally:
            QApplication.restoreOverrideCursor()
            self.btn_rollback.setEnabled(True)

    def update_ai_progress(self, current: int, total: int):
        """Updates the physical QProgressBar from AI thread signal."""
        if total > 0:
            percentage = int((current / total) * 100)
            self.ai_progress_bar.setValue(percentage)

    def on_diam_default_toggled(self, checked: bool):
        if checked:
            self.cb_diam_other.blockSignals(True)
            self.cb_diam_other.setChecked(False)
            self.cb_diam_other.blockSignals(False)
            self.in_custom_diam.setEnabled(False)
            self.in_custom_diam.clear()

    def on_diam_other_toggled(self, checked: bool):
        if checked:
            self.cb_diam_default.blockSignals(True)
            self.cb_diam_default.setChecked(False)
            self.cb_diam_default.blockSignals(False)
            self.in_custom_diam.setEnabled(True)
            self.in_custom_diam.setFocus()
        elif not self.cb_diam_default.isChecked():
            self.cb_diam_default.setChecked(True)


    def on_run_ai_analysis(self):
        proc_dir = self.ai_processed_dir.text().strip().replace('\\', '/')
        well_name = self.ai_well_name.text().strip()
        time = self.ai_time.text().strip()
        mode = self.mode_group.checkedId()

        if not proc_dir or not well_name or not time:
            self.log("❌ [WARNING]: Processed Wells Dir, Well Name, and Time Index must all be filled!")
            return

        try: # get cell diameter
            if self.cb_diam_other.isChecked():
                custom_val = self.in_custom_diam.text().strip()
                if not custom_val:
                    self.log("❌ [WARNING]: Please specify a custom diameter value!")
                    return
                cell_diameter = float(custom_val)
            else:
                cell_diameter = 30.0
        except ValueError:
            self.log("❌ [ERROR]: Custom diameter must be a valid numeric value!")
            return

        # Disable AI button and reset progress to prevent double triggers
        self.btn_run_ai.setEnabled(False)
        self.btn_run_ai.setText("⏳ AI Analysis in Progress...")
        self.ai_progress_bar.setValue(0)


        # Launch background AI worker thread
        self.ai_thread = AIWorkerThread(
            proc_dir=proc_dir,
            well_name=well_name,
            time=time,
            mode=mode,
            model_dir=self.models_dir,
            save_masks=self.cb_save_masks.isChecked(),
            cell_diameter=cell_diameter
        )
        self.ai_thread.log_signal.connect(self.log)
        self.ai_thread.progress_signal.connect(self.update_ai_progress)
        self.ai_thread.finished_signal.connect(self.on_ai_analysis_finished)
        self.ai_thread.error_signal.connect(lambda err: self.log(f"❌ [AI CRITICAL ERROR]: {err}"))
        self.ai_thread.start()

    def on_ai_analysis_finished(self):
        """Restores UI button state upon AI thread completion."""
        self.btn_run_ai.setEnabled(True)
        self.btn_run_ai.setText("🧠 Run AI Segmentation & Update Excel")
        self.log("🏁 [AI PIPELINE]: Task complete.")

    def load_current_mcherry_crop(self):
        if not self.mcherry_files:
            print('No mCherry files.')
            return

        crop_path = self.mcherry_files[
            self.mcherry_current_index
        ]

        try:
            analysis = analyze_mcherry_image(
                crop_path,
                self.mcherry_mask_dir.text().strip(),
                min_diameter=float(
                    self.mcherry_min_diameter.text()
                ),
                max_diameter=float(
                    self.mcherry_max_diameter.text()
                ),
                threshold=float(
                    self.mcherry_threshold.text()
                )
            )

            self.mcherry_current_result = analysis

            self.display_mcherry_analysis(analysis)

        except Exception as e:
            self.log(
                f"[mCherry ERROR]: {crop_path.name}: {e}"
            )

    def initialize_mcherry_analysis(self):
        crop_dir = Path(
            self.mcherry_crop_dir.text().strip()
        )

        if not crop_dir.is_dir():
            self.log("[mCherry ERROR]: Invalid crop directory.")
            return

        self.mcherry_files = sorted(
            crop_dir.glob("*.png")
        )

        if not self.mcherry_files:
            self.log("[mCherry ERROR]: No PNG crops found.")
            return

        self.mcherry_current_index = 0
        self.load_current_mcherry_crop()


    def display_mcherry_analysis(self, analysis):
        crop_path = self.mcherry_files[
            self.mcherry_current_index
        ]

        self.mcherry_current_label.setText(
            f"{crop_path.name} "
            f"({self.mcherry_current_index + 1}/"
            f"{len(self.mcherry_files)})"
        )

        self.mcherry_count_label.setText(
            f"Puncta: {analysis['count']}"
        )

        self.mcherry_figure.clear()

        ax = self.mcherry_figure.add_subplot(111)

        plot_mcherry_analysis(
            ax,
            analysis["image"],
            analysis["mask"],
            analysis["puncta"],
            title=(
                f"{crop_path.name} | "
                f"Puncta: {analysis['count']} | "
                f"Mean intensity: "
                f"{analysis['mean_intensity']:.2f} | "
                f"Mean size: "
                f"{analysis['mean_size']:.2f} px"
            )
        )

        self.mcherry_figure.tight_layout()
        self.mcherry_canvas.draw()

    def replot_current_mcherry(self):
        self.load_current_mcherry_crop()

    def next_mcherry_crop(self):
        if not self.mcherry_files:
            return

        self.save_current_mcherry_visualization()

        if self.mcherry_current_index < len(
            self.mcherry_files
        ) - 1:
            self.mcherry_current_index += 1
            self.load_current_mcherry_crop()

    def previous_mcherry_crop(self):
        if not self.mcherry_files:
            return

        if self.mcherry_current_index > 0:
            self.mcherry_current_index -= 1
            self.load_current_mcherry_crop()

    def save_current_mcherry_visualization(self):
        if not self.mcherry_current_result:
            return

        output_dir = self.mcherry_output_dir.text().strip()

        if not output_dir:
            crop_dir = Path(
                self.mcherry_crop_dir.text().strip()
            )
            output_dir = str(
                crop_dir.parent / "mCherry Analysis"
            )
            self.mcherry_output_dir.setText(output_dir)

        crop_path = self.mcherry_files[
            self.mcherry_current_index
        ]

        output_path = (
            Path(output_dir) /
            "Visualizations" /
            f"{crop_path.stem}_mCherry_puncta.png"
        )

        fig = Figure(figsize=(7, 6))
        ax = fig.add_subplot(111)

        plot_mcherry_analysis(
            ax,
            self.mcherry_current_result["image"],
            self.mcherry_current_result["mask"],
            self.mcherry_current_result["puncta"],
            title=crop_path.name
        )

        save_mcherry_figure(
            fig,
            output_path
        )

        self.log(
            f"[mCherry]: Saved {output_path.name}"
        )

    def run_batch_mcherry_analysis(self):
        crop_dir = self.mcherry_crop_dir.text().strip()
        mask_dir = self.mcherry_mask_dir.text().strip()
        output_dir = self.mcherry_output_dir.text().strip()

        if not output_dir:
            output_dir = str(
                Path(crop_dir).parent /
                "mCherry Analysis"
            )
            self.mcherry_output_dir.setText(output_dir)

        try:
            results = batch_analyze_mcherry(
                crop_dir=crop_dir,
                mask_dir=mask_dir,
                output_dir=output_dir,
                min_diameter=float(
                    self.mcherry_min_diameter.text()
                ),
                max_diameter=float(
                    self.mcherry_max_diameter.text()
                ),
                threshold=float(
                    self.mcherry_threshold.text()
                ),
                progress_callback=self.mcherry_progress_bar.setValue,
                log_callback=self.log
            )

            self.log(
                f"[mCherry]: Batch complete. "
                f"{len(results)} images processed."
            )

        except Exception as e:
            self.log(
                f"[mCherry ERROR]: {e}"
            )

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MicroscopyApp()
    window.show()
    sys.exit(app.exec())