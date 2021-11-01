#
#    Author: David Futschik
#    Provided as part of the Chunkmogrify project, 2021.
#

import sys
import traceback
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '8'
import ctypes
import threading
import setup_cpp_ext
# build extensions
setup_cpp_ext.checked_build()
import resources
# download everything
resources.check_and_download_all()
from time import time
from PIL import Image
from config import global_config

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from qtutil import *
from widgets.workspace import *
from w_directions import known_directions as known_w_directions
from s_presets import known_presets as known_s_presets
from s_presets import limits as s_limits
from styleclip_presets import pretrained_models as known_styleclip_models
from align import unalign_face_npy

from synthesis import SynthesisState, output_modes, edit_modes
from mask import MaskState
from widgets.mask_painter import PainterWidget as MaskPainter
from widgets.editor import MultiEditWidget

stringres = dotdict({
    'assert_fail': 'General Failure:',
    'title': 'Chunkmogrify Editor', 
})
s = stringres

def rename(str):
    subparts = str.split('_')
    for i in range(len(subparts)):
        subparts[i] = subparts[i][0:1].upper() + subparts[i][1:]
    return ' '.join(subparts)

def unrename(str):
    str = str.replace(' ', '_')
    return str.lower()

status = None
class Chunkmogrify(QMainWindow):
    refresh = pyqtSignal()
    error_handle = pyqtSignal(str, str)
    def __init__(self, app_ref):
        super().__init__()

        self.elements = dotdict({})
        self.status = QLabel("Empty")
        self.app_ref = app_ref

        # Don't change the order of these initializations
        imsize = global_config().generator_native_resolution # h, w
        self.max_segments = global_config().max_segments
        self.mask_painter = MaskPainter(*imsize, self.max_segments, self.status.setText)
        self.workspace = WorkspaceWidget(self, initial_image=None, forward_widgets=[self.mask_painter])
        self.mask_state = MaskState(*imsize, self.max_segments, [self.workspace.set_overlay, self.mask_painter.rewrite_with_numpy_mask])
        self.mask_painter.update_callbacks += [self.mask_state.set_to]
        self.synthesis = SynthesisState(self.workspace.set_content, 
                                            self.mask_state.set_to, 
                                            self.mask_state.numpy_buffer, 
                                            self.mask_state.get_mask_version,
                                            self.projection_start,
                                            self.projection_end)
        self.mask_state.update_callbacks += [self.mask_painter.rewrite_with_numpy_mask, self.synthesis.on_mask_changed]

        self.mask_painter.toggle(False) # Starts disabled.
        self.build_interface()
        self.error_handle.connect(lambda x,y: notify_user_error(x, y))

        self.status.setText("Ready")
        self._post_init()

    def set_status(self, msg):
        self.status.setText(msg)
        self.app_ref.processEvents()

    def build_interface(self):
        w = QWidget()
        layout = self.build_main_layout()
        w.setLayout(layout)
        self.setMinimumHeight(900)
        self.setMinimumWidth(1500)
        self.setCentralWidget(w)
        self.make_menu()
        self.show()

    def build_main_layout(self):
        vbox = QVBoxLayout()
        vbox.addLayout(self.build_top_navigation())
        vbox.addWidget(QHLine())
        vbox.addLayout(self.build_middle())
        vbox.addWidget(QHLine())
        vbox.addLayout(self.build_bottom_menu())
        vbox.addStretch(1)
        return vbox

    def make_menu(self):
        self.elements.menubar = self.menuBar()
        self.make_file_menu()
        self.make_tools_menu()
        self.make_import_menu()
        self.make_export_menu()
        if global_config().show_debug_menu:
            self.make_debug_menu()

    def make_import_menu(self):
        self.elements.menubar.imports = self.elements.menubar.addMenu("&Import")

        import_masks_action = QAction("Import mask", self)
        def import_masks():
            path = QFileDialog(self).getExistingDirectory(self, "Select directory to import masks")
            if path == "":
                # Cancel
                return
            self.mask_state.load_masks(source_dir=os.path.join(path)) # global_config().import_directory, 'masks/'
            self.workspace.update()
        import_masks_action.triggered.connect(import_masks)

        import_ws_action = QAction("Import Ws", self)
        def import_ws():
            path = QFileDialog(self).getExistingDirectory(self, "Select directory to import W")
            if path == "":
                # Cancel
                return
            self.synthesis.load_ws(source_dir=os.path.join(path)) # ws/
        import_ws_action.triggered.connect(import_ws)

        self.elements.menubar.imports.import_masks_action = import_masks_action
        self.elements.menubar.imports.addAction(import_masks_action)
        self.elements.menubar.imports.import_ws_action = import_ws_action
        self.elements.menubar.imports.addAction(import_ws_action)

    def make_export_menu(self):
        self.elements.menubar.export = self.elements.menubar.addMenu("&Export")
        
        export_masks_action = QAction("Export mask", self)
        def export_masks():
            path = QFileDialog(self).getExistingDirectory(self, "Select directory to export masks")
            if path == "":
                # Cancel
                return
            prefix = os.path.join(path, 'masks/')
            # Good to check visually if the masks are correct
            make_dirs_if_not_exists(prefix)
            self.mask_state.save_masks(target_dir=prefix, painter=self.mask_painter, max_segments=self.max_segments)
            self.set_status(f"Masks exported to {prefix}")
        export_masks_action.triggered.connect(export_masks)

        export_projections_action = QAction("Export projections", self)
        def export_projections():
            path = QFileDialog(self).getExistingDirectory(self, "Select directory to export projections")
            if path == "":
                # Cancel
                return
            # Export projections, should move it from synth to here.
            prefix=os.path.join(path, 'projections/')
            make_dirs_if_not_exists(prefix)
            self.synthesis.export_projections(prefix)
            self.set_status(f"Projections exported to {prefix}")
        export_projections_action.triggered.connect(export_projections)

        export_w_action = QAction("Export projections W", self)
        def export_w():
            path = QFileDialog(self).getExistingDirectory(self, "Select directory to export W")
            if path == "":
                # Cancel
                return
            prefix=os.path.join(path, 'ws/')
            make_dirs_if_not_exists(prefix)
            self.synthesis.save_ws(target_dir=prefix)
            self.set_status(f"Ws exported to {prefix}")
        export_w_action.triggered.connect(export_w)

        # TODO: Export input with edits applied.

        self.elements.menubar.export.export_masks_action = export_masks_action
        self.elements.menubar.export.export_projections_action = export_projections_action
        self.elements.menubar.export.export_w_action = export_w_action

        self.elements.menubar.export.addAction(export_masks_action)
        self.elements.menubar.export.addAction(export_projections_action)
        self.elements.menubar.export.addAction(export_w_action)

    def make_file_menu(self):
        self.elements.menubar.file = self.elements.menubar.addMenu("&File")

        image_from_file = QAction("Image from File", self)

        is_first_time = True
        def open_image_file():
            nonlocal is_first_time
            path, filt = QFileDialog(self).getOpenFileName(self, "Open Image", filter="Images (*.png *.jpg *.jpeg)")
            if path == "":
                # Cancel
                return
            
            self.set_status("Loading image..")
            print(f'Loading {path}')

            exception_raised = False
            def load_and_align_image(path):
                try:
                    self.synthesis.load_image_and_reset(path, align=not global_config().skip_alignment)
                except RuntimeError as _:
                    nonlocal exception_raised
                    exception_raised = True
                get_notify_wait().release.emit()

            t = threading.Thread(target=load_and_align_image, args=(path,))
            t.start()
            # Could happen that release is called before acquire, previously I solved that
            # with a trap thread, but this shouldn't happen here for a long running task.
            get_notify_wait().acquire.emit("Loading image..")
            if exception_raised:
                notify_user_error("Face detector", f"No faces found in {path}", self)
                return
            self.elements.imlabel.setText(path)
            if is_first_time:
                is_first_time = False
                self.projection_enable()
                self.mask_painter.toggle(True)
                self.elements.enable_painting.setEnabled(True)
                self.elements.show_mask.setEnabled(True)
            self.status.setText("Ready")

        image_from_file.triggered.connect(open_image_file)

        self.elements.menubar.file.addAction(image_from_file)

    def make_tools_menu(self):
        self.elements.menubar.tools = self.elements.menubar.addMenu("&Tools")
        
        # TODO: Extend multiplier

        mask_snap_action = QAction("Refine mask")
        def mask_snap():
            self.synthesis.mask_refine( dA=0.001, dB=0.1, iters=300 )
        mask_snap_action.triggered.connect(mask_snap)

        self.elements.menubar.tools.mask_snap_action = mask_snap_action
        self.elements.menubar.tools.addAction(mask_snap_action)

    def make_debug_menu(self):
        debug_menu = self.elements['menubar'].addMenu("&Debug")
        self.elements.menubar.debug = debug_menu
        self.elements['menu.reload_action'] = QAction("&Reload", self)
        self.elements['menu.reload_action'].setShortcut("CTRL+R")
        self.elements['menu.reload_action'].triggered.connect(lambda: os.execv(sys.executable, ['python'] + sys.argv))

        self.elements.menubar.debug.save_last_output = QAction("Save last output", self)
        def save_last_output():
            make_dirs_if_not_exists('exports/edited_projection')
            for idx in range(len(self.synthesis.last_output)):
                res_np = self.synthesis.last_output[idx]
                Image.fromarray(res_np).save(os.path.join('exports/edited_projection', f'{idx:02d}.png'))
        self.elements.menubar.debug.save_last_output.triggered.connect(save_last_output)

        self.elements.menubar.debug.save_edits = QAction("Save edits", self)
        def save_edits():
            edits = self.elements.editor.gather_values()
            import pickle
            with open(os.path.join(global_config().export_directory, 'edits.txt')) as f:
                pickle.dump(edits, f)
        self.elements.menubar.debug.save_edits.triggered.connect(save_edits)

        debug_menu.addAction(self.elements['menu.reload_action'])
        debug_menu.addAction(self.elements.menubar.debug.save_last_output)
        debug_menu.addAction(self.elements.menubar.debug.save_edits)

    def build_top_navigation(self):
        hbox = QHBoxLayout()
        title_label = QLabel(s.title)
        title_label.setStyleSheet("font-size: 22px; font: bold;")
        hbox.addStretch(1)
        center_layout = QVBoxLayout()
        hbox.addLayout(center_layout)
        hbox.addStretch(1)
        return hbox 

    def build_middle(self):
        hbox = QHBoxLayout()
        hbox.addLayout(self.build_middle_left(), 15)
        hbox.addWidget(QVLine())
        hbox.addLayout(self.build_middle_workspace(), 50)
        hbox.addWidget(QVLine())
        vbox = QVBoxLayout()

        ar1 = KeepArWidget()
        self.side_image_1 = WorkspaceWidget(self, initial_image='resources/iconS.png', forward_widgets=[])
        self.side_image_1.inset_border(1, (0, 0, 0))
        self.side_image_1.set_scale_factor(0.1)
        ar1.set_ar(1)
        ar1.set_widget(self.side_image_1)

        ar2 = KeepArWidget()
        self.side_image_2 = WorkspaceWidget(self, initial_image='resources/iconS.png', forward_widgets=[])
        self.side_image_2.inset_border(1, (0, 0, 0))
        self.side_image_2.set_scale_factor(0.1)
        ar2.set_ar(1)
        ar2.set_widget(self.side_image_2)

        self.synthesis.set_unaltered_output(self.side_image_1.set_content)
        self.synthesis.set_difference_output(self.side_image_2.set_content)

        vbox.addWidget(QLabel("Original Image"))
        vbox.addWidget(ar1)
        vbox.addWidget(QLabel("Difference Map"))
        vbox.addWidget(ar2)
        vbox.addStretch(100)
        hbox.addLayout(vbox)
        return hbox

    def build_middle_left(self):
        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addLayout(self.build_controls())
        vbox.addLayout(hbox)

        lowerhbox = QHBoxLayout()
        # This makes the whole panel stretch with width.
        lowerhbox.addStretch()
        vbox.addLayout(lowerhbox)
        return vbox

    def build_middle_workspace(self):
        vbox = QVBoxLayout()
        grid = QGridLayout()

        workspace_ar = KeepArWidget()
        workspace_ar.set_widget(self.workspace)
        workspace_ar.set_ar(1)

        grid.addWidget(workspace_ar)

        vbox.addLayout(grid)
        self.elements.imlabel = QLabel("")
        self.elements.general_progress_bar = QProgressBar()
        hbox = QHBoxLayout()
        self.elements.general_progress_bar = QProgressBar()
        self.elements.general_progress_bar.setMaximumHeight(15)
        hbox.addWidget(self.elements.general_progress_bar)
        hbox.addStretch()
        hbox.addWidget(self.elements.imlabel)
        vbox.addLayout(hbox)
        return vbox

    def build_bottom_menu(self):
        hbox = QHBoxLayout()
        hbox.addWidget(self.status)
        return hbox

    def build_controls(self):
        self.parameter_list = QVBoxLayout()
        vbox = self.parameter_list
        self.elements.debug_line = QLabel("")

        self.elements.enable_painting = QCheckBox("Mask Paint E&nabled")
        self.elements.show_mask = QCheckBox("Show &Mask")

        self.elements.output_mode_label = QLabel("Output mode")
        self.elements.output_mode_combo = QComboBox()

        self.elements.editor = MultiEditWidget()

        self.elements.proj_layout = QGridLayout()
        self.elements.proj_count_selector = QSpinBox()
        self.elements.proj_count_selector.setMaximum(500)
        self.elements.proj_count_selector.setValue(100)
        self.elements.proj_button = QPushButton("Projection Steps")
        self.elements.proj_forward_button = QPushButton("Forward Projection")
        self.elements.stop_projection_button = QPushButton("Stop Projection")

        self.elements.proj_layout.addWidget(self.elements.proj_count_selector, 0, 0, 1, 1)
        self.elements.proj_layout.addWidget(self.elements.proj_button, 0, 1, 1, 2)
        # Ready to implement a forward pass latent.
        # self.elements.proj_layout.addWidget(self.elements.proj_forward_button, 1, 0, 1, 3)
        self.elements.proj_layout.addWidget(self.elements.stop_projection_button, 2, 0, 1, 3)

        self.elements.pti_run_button = QPushButton("Iterations of Pivotal Tuning")
        self.elements.pti_count_selector = QSpinBox()
        self.elements.pti_count_selector.setMaximum(350)
        self.elements.pti_count_selector.setValue(50)
        self.elements.pti_layout = QGridLayout()

        self.elements.pti_layout.addWidget(self.elements.pti_count_selector, 0, 0, 1, 1)
        self.elements.pti_layout.addWidget(self.elements.pti_run_button, 0, 1, 1, 2)

        self.elements.freeze_button = QPushButton("Freeze Image")
        self.elements.reset_button = QPushButton("Reset Projector")
        self.elements.save_current = QPushButton("Save Current Image")
        self.elements.save_current_unaligned = QCheckBox("Unalign")
        self.elements.save_current_unaligned.setChecked(True)
        self.elements.save_layout = QGridLayout()
        self.elements.save_layout.addWidget(self.elements.save_current, 0, 0, 1, 1)
        self.elements.save_layout.addWidget(self.elements.save_current_unaligned, 0, 1, 1, 1)


        vbox.addWidget(QLabel("Painting"))
        vbox.addWidget(self.elements.show_mask)
        vbox.addWidget(self.elements.enable_painting)
        vbox.addWidget(QHLine())
        vbox.addWidget(QLabel("Projection"))
        vbox.addLayout(self.elements.proj_layout)
        vbox.addWidget(self.elements.output_mode_label)
        vbox.addWidget(self.elements.output_mode_combo)
        vbox.addWidget(QHLine())
        vbox.addWidget(QLabel("Attribute Editor"))
        vbox.addWidget(self.elements.editor)
        vbox.addWidget(QHLine())
        vbox.addWidget(self.elements.freeze_button)
        vbox.addWidget(self.elements.reset_button)
        
        vbox.addWidget(QHLine())
        vbox.addWidget(QLabel("Save Image"))
        vbox.addLayout(self.elements.save_layout)
        vbox.addWidget(QHLine())
        vbox.addWidget(QLabel("Miscellaneous"))
        vbox.addLayout(self.elements.pti_layout)
        
        # Stretch given to editor widget now. Otherwise: vbox.addStretch()
        return vbox

    def projection_enable(self):
        self.elements.proj_count_selector.setEnabled(True)
        self.elements.proj_button.setEnabled(True)
        self.elements.save_current.setEnabled(True)
        self.elements.save_current_unaligned.setEnabled(True)
        self.elements.menubar.imports.import_masks_action.setEnabled(True)
        self.elements.menubar.export.export_masks_action.setEnabled(True)

    def projection_start(self):
        self.elements.stop_projection_button.setEnabled(True)
        self.elements.reset_button.setEnabled(True)
        self.elements.freeze_button.setEnabled(True)
        self.elements.pti_run_button.setEnabled(True)
        self.elements.pti_count_selector.setEnabled(True)
        self.elements.menubar.export.export_projections_action.setEnabled(True)
        self.elements.menubar.export.export_w_action.setEnabled(True)
        self.elements.menubar.imports.import_ws_action.setEnabled(True)
        self.elements.menubar.tools.mask_snap_action.setEnabled(True)
        self.elements.editor.setEnabled(True)

    def projection_end(self):
        self.elements.stop_projection_button.setEnabled(False)
        self.elements.reset_button.setEnabled(False)
        self.elements.freeze_button.setEnabled(False)
        self.elements.pti_run_button.setEnabled(False)
        self.elements.pti_count_selector.setEnabled(False)
        self.elements.menubar.export.export_projections_action.setEnabled(False)
        self.elements.menubar.export.export_w_action.setEnabled(False)
        self.elements.menubar.tools.mask_snap_action.setEnabled(False)
        self.elements.menubar.imports.import_ws_action.setEnabled(False)
        self.elements.editor.setEnabled(False)

    def _post_init(self):

        self.elements.show_mask.setEnabled(False)
        self.elements.enable_painting.setEnabled(False)
        self.elements.general_progress_bar.hide()

        self.elements.proj_count_selector.setEnabled(False)
        self.elements.proj_button.setEnabled(False)
        self.elements.save_current.setEnabled(False)
        self.elements.save_current_unaligned.setEnabled(False)

        self.elements.stop_projection_button.setEnabled(False)
        self.elements.reset_button.setEnabled(False)
        self.elements.freeze_button.setEnabled(False)
        self.elements.pti_run_button.setEnabled(False)
        self.elements.pti_count_selector.setEnabled(False)
        self.elements.editor.setEnabled(False)

        self.elements.menubar.export.export_projections_action.setEnabled(False)
        self.elements.menubar.export.export_w_action.setEnabled(False)
        self.elements.menubar.tools.mask_snap_action.setEnabled(False)
        self.elements.menubar.export.export_masks_action.setEnabled(False)
        self.elements.menubar.imports.import_masks_action.setEnabled(False)
        self.elements.menubar.imports.import_ws_action.setEnabled(False)


        def reset():
            Thread(target=lambda: self.synthesis.reset_projection()).start()
        self.elements.reset_button.clicked.connect(reset)

        # Disable / enable painting the mask.
        self.elements.enable_painting.setChecked(self.mask_painter.enabled())
        def toggle_painting(s):
            self.mask_painter.toggle(s)
            if s: self.elements.show_mask.setChecked(True)
            self.workspace.update()
        self.elements.enable_painting.stateChanged.connect(lambda s: toggle_painting(s))
        # Show / Hide the mask overlay.
        self.elements.show_mask.setChecked(self.workspace.overlay_enabled)
        # For technical reasons there are 2 overlays, 1 in workspace and 1 in mask painter.
        def toggle_overlays(s):
            self.workspace.toggle_overlay(s)
            if s: self.mask_painter.show()
            else:
                # Disable painting too.
                self.mask_painter.hide()
                self.elements.enable_painting.setChecked(False)
        self.elements.show_mask.stateChanged.connect(lambda s: toggle_overlays(s))

    
        # Freeze button freezes the current image and resets projector.
        def freeze():
            current_image = self.workspace.current_image_as_numpy()
            # Workspace includes alphamap, which is apparently garbage data.
            Thread(target=
                lambda: self.synthesis.set_image_and_reset(current_image[:, :, 0:3])
            ).start()
        self.elements.freeze_button.clicked.connect(freeze)

        # Projection tells synthesis to synthesize with a step.
        def step():
            self.synthesis.synthesize_with_project_step()

        stop_token = False
        def set_stop():
            nonlocal stop_token
            stop_token = True

        class Stepper(QObject):
            step = pyqtSignal()
        
        self.stepper = Stepper()
        self.stepper.step.connect(lambda: step())

        class UpdateStatus(QObject):
            update = pyqtSignal(str)
        class UpdateProgressBar(QObject):
            start = pyqtSignal(int)
            update = pyqtSignal(int)
            finished = pyqtSignal()

        update_status_object = UpdateStatus()
        update_status_object.update.connect(lambda msg: self.set_status(msg))
        update_progress_object = UpdateProgressBar()
        def progress_start(max):
            self.elements.general_progress_bar.setMaximum(max)
            self.elements.general_progress_bar.setValue(0)
            self.elements.general_progress_bar.show()
        def progress_end():
            self.elements.general_progress_bar.hide()
        update_progress_object.start.connect(lambda max: progress_start(max))
        update_progress_object.finished.connect(progress_end)
        update_progress_object.update.connect(lambda i: self.elements.general_progress_bar.setValue(i))

        def steps_(amount):
            nonlocal stop_token
            stop_token = False
            update_status_object.update.emit(f"Working..")
            update_progress_object.start.emit(amount)
            for i in range(amount):
                if stop_token: break
                step()
                update_status_object.update.emit(f"Working.. Iteration = {i + 1}")
                update_progress_object.update.emit(i)
                # This runs the event loop to update widgets
                # self.app_ref.processEvents()
            # Because the result shown comes from before the last gradient update, need to run once more
            self.synthesis._synthesize_with_last_params()
            update_status_object.update.emit(f"Ready")
            update_progress_object.finished.emit()

        def profile_steps(amount):
            import cProfile
            p = cProfile.Profile()
            p.runcall(steps_, amount)
            p.dump_stats("trace.perf")
            # snakeviz for viz

        def run_in_thread(fn, *args, done_callback=None):
            def run_impl():
                fn(*args)
                if done_callback: done_callback()
            threading.Thread(target=run_impl).start()

        def project():
            self.elements.proj_count_selector.setEnabled(False)
            self.elements.proj_button.setEnabled(False)
            self.elements.stop_projection_button.setEnabled(True)
            def enable_again():
                self.elements.proj_count_selector.setEnabled(True)
                self.elements.proj_button.setEnabled(True)
                self.elements.stop_projection_button.setEnabled(False)
            run_in_thread(steps_, self.elements.proj_count_selector.value(),
                    done_callback=enable_again)

        self.elements.proj_button.clicked.connect(lambda s: project())
        self.elements.stop_projection_button.clicked.connect(lambda s: set_stop())

        output_mode_mapping = {}
        for mode in output_modes:
            output_mode_mapping[rename(mode)] = mode

        # Tell synthesis to show output mode
        for pretty_mode in output_mode_mapping.keys():
            self.elements.output_mode_combo.addItem(pretty_mode)
        # Preset "masked"
        self.elements.output_mode_combo.setCurrentIndex(2)
        self.elements.output_mode_combo.currentTextChanged.connect(
            lambda t: self.synthesis.set_output_mode(output_mode_mapping[t])
        )

        # Save and unalign
        def save_unaligned():
            aligned = self.workspace.current_image_as_numpy()[:, :, :3]
            # TODO: move the storage of alignment params
            alignment_params = self.synthesis.alignment_params
            unaligned_npy = unalign_face_npy(aligned, alignment_params)
            return export_image(global_config().export_directory, unaligned_npy)
        # Save current shown image (whatever the result is)
        def save():
            if self.elements.save_current_unaligned.isChecked():
                saved_at = save_unaligned()
            else:
                saved_at = export_image(global_config().export_directory, self.workspace.get_current(with_overlay=False))
            update_status_object.update.emit(f"Saved image at {saved_at}")

        self.elements.save_current.clicked.connect(
            lambda s: save()
        )

        self.elements.proj_forward_button.clicked.connect(
            lambda s: Thread(target=self.synthesis.forward_projection).start()
        )

        # Fill directions from dict defined in stylegan_project.
        direction_mapping = {}
        for name in known_w_directions().keys():
            direction_mapping[rename(name)] = name

        edit_mode_mapping = {}
        for mode in edit_modes:
            edit_mode_mapping[rename(mode)] = mode

        self.elements.editor.set_w_directions(direction_mapping.keys())

        presets = {rename(k): v for k, v in known_s_presets().items()}

        self.elements.editor.set_s_presets(presets)
        self.elements.editor.set_s_limits(s_limits()['layer'], s_limits()['channel'])

        models = {rename(k): v for k, v in known_styleclip_models().items()}
        self.elements.editor.set_styleclip_models([k for k, v in models.items()])

        choices = ['all'] + [str(i) for i in range(global_config().max_segments)]
        self.elements.editor.set_segment_selection(choices)

        def attribute_update(list_of_values):
            # translate strings
            for params in list_of_values:
                params.type = unrename(params.type)
                for k, v in params.parameters.items():
                    if type(v) == str: params.parameters[k] = unrename(v)
            self.synthesis.synthesize_with_params(list_of_values)
        self.elements.editor.set_push_function(attribute_update)

        # PTI integration
        def pti_run():
            def pti_steps(num):
                update_progress_object.start.emit(num)
                for it in range(num):
                    self.synthesis.pti_step()
                    update_status_object.update.emit(f"Working.. Step {it}")
                    update_progress_object.update.emit(it)
                update_status_object.update.emit(f"Ready")
                update_progress_object.finished.emit()
            run_in_thread(pti_steps, self.elements.pti_count_selector.value())
            
        def pti_warning():
            response = QMessageBox.question(self, "Warning", 
                "Running Pivotal Tuning depends on the current state of projection, "\
                "WILL change the current model (not on disk) and can take some time. Continue?")
            if response != QMessageBox.Yes:
                return
            pti_run()
        self.elements.pti_run_button.clicked.connect(lambda s: pti_warning())

    def shutdown(self):
        pass

    def window_size_rat(self, pct):
        return self.height() * pct, self.width() * pct

app = None
def run():

    # set up correct appid (required for python to be able to have a nonstandard
    # taskbar icon under windows)
    t0 = time()
    appid = 'srp.futscdav.gmail.com'
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(appid)
    except:
        pass # Not implemented on other OS

    global app
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('resources/iconS.png'))
    try:
        gui = Chunkmogrify(app)
        set_default_parent(gui)
        gui.setWindowTitle(s.title)
    except Exception as e:
        tb = sys.exc_info()[2]
        traceback.print_tb(tb)
        print(f'{type(e).__name__} {e}')
        notify_user_error(s.assert_fail, f'{e.__class__.__name__}: {str(e)}')
        sys.exit(1)

    print(f"Gui init: {time() - t0:.2f}s")

    def excepthook(exc_type, exc_value, exc_traceback):
        tb = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        notify_user_error(s.assert_fail, f'{exc_value.__class__.__name__}: {str(exc_value)}\n\n\n{tb}')
        print(tb)
        app.exit(1)

    sys.excepthook = excepthook

    errcode = app.exec_()
    gui.shutdown()
    sys.exit(errcode)

if __name__ == '__main__':
    run()