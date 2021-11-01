#
#    Author: David Futschik
#    Provided as part of the Chunkmogrify project, 2021.
#

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from qtutil import *

_widget_params = dotdict({
    'w_edit': dotdict(),
    's_edit': dotdict(),
    'styleclip_edit': dotdict(),
})

# Gets rid of the annoying trailing behavior.
class QJumpSlider(QSlider):
    def __init__(self, parent = None):
        super().__init__(parent)
     
    def mousePressEvent(self, event):
        # Jump to click position.
        self.setValue(QStyle.sliderValueFromPosition(
            self.minimum(), self.maximum(), event.x(), self.width())
        )
     
    def mouseMoveEvent(self, event):
        # Jump to pointer position while moving.
        self.setValue(QStyle.sliderValueFromPosition(
            self.minimum(), self.maximum(), event.x(), self.width())
        )

class QComboBoxWithoutWheel(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.installEventFilter(self)

    def eventFilter(self, widget, event):
        super_blocks = super().eventFilter(widget, event)
        if not super_blocks and event.type() == QEvent.Wheel:
            return True
        return super_blocks


class ReplaceableWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.content_container = QVBoxLayout()
        self.content = QVBoxLayout()
        self.holds_layout = True
        self.setLayout(self.content_container)

    def _remove_all(self):
        if self.holds_layout:
            self.content_container.removeItem(self.content)
            # To properly "hide" the elements, I need to hide them individually, because their parent is the main window
            for item in [self.content.itemAt(i).widget() for i in range(self.content.count())]:
                item.setVisible(False)
        else:
            self.content_container.removeWidget(self.content)
            self.content.setVisible(False)
        self.update()

    def set_content_layout(self, new_layout):
        self._remove_all()
        self.content_container.addLayout(new_layout)
        self.content = new_layout
        self.holds_layout = True
        # Show if they were hidden
        for item in [self.content.itemAt(i).widget() for i in range(self.content.count())]:
            item.setVisible(True)
        self.update()

    def set_content_widget(self, new_widget):
        self._remove_all()
        self.content_container.addWidget(new_widget)
        self.content = new_widget
        self.content.setVisible(True)
        self.holds_layout = False
        self.update()


class None_EditWidget(QWidget):
    def __init__(self, on_change, parent=None):
        super().__init__(parent=parent)
        self.setLayout(QVBoxLayout())

    def current_values(self):
        return dotdict()

class StyleClip_EditWidget(QWidget):
    def __init__(self, on_change, parent=None):
        super().__init__(parent=parent)
        self.setLayout(QVBoxLayout())
        self.on_change = on_change

        self.model_combo_label = QLabel("Model:")
        self.model_combo = QComboBox()
        
        self.strength_slider_label = QLabel("Strength: 0.00")
        self.strength_slider = QJumpSlider(1)
        self.strength_slider.setMaximum(102)
        self.strength_slider.setTickInterval(50)
        self.strength_slider.setValue(51)

        self.layout().addWidget(self.model_combo_label)
        self.layout().addWidget(self.model_combo)
        self.layout().addWidget(self.strength_slider_label)
        self.layout().addWidget(self.strength_slider)
        self._connect_behavior()

    def _connect_behavior(self):
        def on_model_change(new_model):
            self._trigger_value_change()
        def on_strength_change(new_strength):
            real_new_val = (new_strength - 51) / self.strength_slider.maximum() * 2
            real_new_val = real_new_val / 4 # set it to [-0.25, 0.25]
            self.strength_slider_label.setText(f"Strength: {real_new_val:.2f}")
            self._trigger_value_change()
        self.strength_slider.valueChanged.connect(on_strength_change)
        self.model_combo.currentTextChanged.connect(on_model_change)

        for model in sorted(_widget_params.styleclip_edit.models):
            self.model_combo.addItem(model)

    def _trigger_value_change(self):
        self.on_change()

    def current_values(self):
        values = dotdict()
        values.model = self.model_combo.currentText()
        values.strength = (self.strength_slider.value() - 51) / self.strength_slider.maximum() * 2
        values.strength = values.strength / 4 # set it to [-0.25, 0.25]
        return values

class W_EditWidget(QWidget):
    def __init__(self, on_change, parent=None):
        super().__init__(parent=parent)
        self.setLayout(QVBoxLayout())
        self.on_change = on_change

        self.direction_combo_label = QLabel("Direction:")
        self.direction_combo = QComboBox()

        self.direction_slider_label = QLabel("Direction value: 0.00")
        self.direction_slider = QJumpSlider(1)
        self.direction_slider.setMaximum(102)
        self.direction_slider.setTickInterval(50)
        self.direction_slider.setValue(51)

        self.multiplier_slider_label = QLabel("Multiplier value: 1")
        self.multiplier_slider = QJumpSlider(1)
        self.multiplier_slider.setTickPosition(20)
        self.multiplier_slider.setValue(1)
        self.multiplier_slider.setMaximum(20)

        self.layout().addWidget(self.direction_combo_label)
        self.layout().addWidget(self.direction_combo)
        self.layout().addWidget(self.direction_slider_label)
        self.layout().addWidget(self.direction_slider)
        self.layout().addWidget(self.multiplier_slider_label)
        self.layout().addWidget(self.multiplier_slider)
        self._connect_behavior()
    
    def _connect_behavior(self):
        def on_dir_change(new_dir):
            self._trigger_value_change()
        def on_val_change(new_val):
            real_new_val = (new_val - 51) / self.direction_slider.maximum() * 2
            self.direction_slider_label.setText(f"Direction value: {real_new_val:.2f}")
            self._trigger_value_change()
        def on_mult_change(new_val):
            self.multiplier_slider_label.setText(f"Multiplier value: {new_val}")
            self._trigger_value_change()

        self.direction_combo.currentTextChanged.connect(on_dir_change)
        self.direction_slider.valueChanged.connect(on_val_change)
        self.multiplier_slider.valueChanged.connect(on_mult_change)

        for direction in sorted(_widget_params.w_edit.directions):
            self.direction_combo.addItem(direction)

    def _trigger_value_change(self):
        self.on_change()

    def current_values(self):
        values = dotdict()
        values.direction = self.direction_combo.currentText()
        values.value = (self.direction_slider.value() - 51) / self.direction_slider.maximum() * 2
        values.multiplier = self.multiplier_slider.value()
        return values


class S_EditWidget(QWidget):
    def __init__(self, on_change, parent=None):
        super().__init__(parent=parent)
        self.setLayout(QVBoxLayout())
        self.on_change = on_change

        self.preset_combo_label = QLabel("Preset")
        self.preset_combo = QComboBoxWithoutWheel()

        self.layer_label = QLabel("Layer Index")
        self.layer_input = QSpinBox()
        self.channel_label = QLabel("Channel Index")
        self.channel_input = QSpinBox()

        self.direction_slider_label = QLabel("Direction value: 0.00")
        self.direction_slider = QJumpSlider(1)
        self.direction_slider.setTickInterval(50)
        self.direction_slider.setMaximum(102)
        self.direction_slider.setValue(51)

        self.multiplier_slider_label = QLabel("Multiplier value: 1")
        self.multiplier_slider = QJumpSlider(1)
        self.multiplier_slider.setTickPosition(20)
        self.multiplier_slider.setValue(1)
        self.multiplier_slider.setMaximum(100)

        self.input_layout_top = QHBoxLayout()
        self.input_layout_1 = QVBoxLayout()
        self.input_layout_2 = QVBoxLayout()

        self.input_layout_1.addWidget(self.layer_label)
        self.input_layout_1.addWidget(self.layer_input)
        self.input_layout_2.addWidget(self.channel_label)
        self.input_layout_2.addWidget(self.channel_input)
        self.input_layout_top.addLayout(self.input_layout_1)
        self.input_layout_top.addLayout(self.input_layout_2)

        self.layout().addWidget(self.preset_combo_label)
        self.layout().addWidget(self.preset_combo)
        self.layout().addLayout(self.input_layout_top)
        self.layout().addWidget(self.direction_slider_label)
        self.layout().addWidget(self.direction_slider)
        self.layout().addWidget(self.multiplier_slider_label)
        self.layout().addWidget(self.multiplier_slider)
        self._connect_behavior()

    def _connect_behavior(self):
        def on_val_change(new_val):
            real_new_val = (new_val - 51) / self.direction_slider.maximum() * 2
            self.direction_slider_label.setText(f"Direction value: {real_new_val:.2f}")
            self._trigger_value_change()
        def on_mult_change(new_val):
            self.multiplier_slider_label.setText(f"Multiplier value: {new_val}")
            self._trigger_value_change()
        def on_input_channel_change(_):
            self._trigger_value_change()
        def on_input_layer_change(new_layer):
            self.channel_input.setMaximum(_widget_params.s_edit.limits.channel_limit[new_layer] - 1)
            self._trigger_value_change()
        self.direction_slider.valueChanged.connect(on_val_change)
        self.multiplier_slider.valueChanged.connect(on_mult_change)
        self.layer_input.valueChanged.connect(on_input_layer_change)
        self.channel_input.valueChanged.connect(on_input_channel_change)

        self.layer_input.setMaximum(_widget_params.s_edit.limits.layer_limit - 1)
        self.channel_input.setMaximum(_widget_params.s_edit.limits.channel_limit[0] - 1)

        self.preset_combo.addItem("Custom")
        for preset_name in sorted(list(_widget_params.s_edit.presets.keys())):
            self.preset_combo.addItem(preset_name)
        
        def set_preset(new_text):
            is_custom = new_text == "Custom"
            if is_custom:
                self.layer_input.setEnabled(True)
                self.channel_input.setEnabled(True)
            else:
                self.layer_input.setEnabled(False)
                self.channel_input.setEnabled(False)
                preset_params = _widget_params.s_edit.presets[new_text]
                self.layer_input.blockSignals(True)
                self.channel_input.blockSignals(True)
                self.layer_input.setValue(preset_params.layer)
                self.channel_input.setValue(preset_params.channel)
                self.layer_input.blockSignals(False)
                self.channel_input.blockSignals(False)
            self._trigger_value_change()
        self.preset_combo.currentTextChanged.connect(set_preset)

    def _trigger_value_change(self):
        self.on_change()

    def current_values(self):
        values = dotdict()
        values.layer = self.layer_input.value()
        values.channel = self.channel_input.value()
        values.value = (self.direction_slider.value() - 51) / self.direction_slider.maximum() * 2
        values.multiplier = self.multiplier_slider.value()
        return values


class EditWidget(QFrame):
    def __init__(self, idx, on_change, on_remove, parent=None) -> None:
        super().__init__(parent=parent)
        self.on_change = on_change
        self.on_remove = on_remove
        self.idx = idx

        self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)
        self._build_widget()
        self.current_widget = self.type_choices['None']

    def _switch_mode(self, new_mode):
        self.current_widget = self.type_choices[new_mode]
        self.edit_content.set_content_widget(self.current_widget)
        self.on_change()

    def _build_widget(self):
        self.type_combo = QComboBoxWithoutWheel()

        self.segment_selection_combo = QComboBoxWithoutWheel()
        for choice in _widget_params.segment_selection.choices:
            self.segment_selection_combo.addItem(choice)

        self.enabled_checkbox = QCheckBox("Enabled")
        self.enabled_checkbox.setChecked(True)
        self.enabled_checkbox.stateChanged.connect(lambda s: self.on_change())

        self.remove_self_button = QToolButton()
        self.remove_self_button.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxCritical)) # QIcon.fromTheme("list-remove")
        self.remove_self_button.clicked.connect(self.on_remove)

        self.ctrl_layout = QHBoxLayout()
        self.ctrl_layout.addWidget(QLabel(f'Editor {self.idx}'))

        if len(_widget_params.segment_selection.choices) > 2:
            self.ctrl_layout.addWidget(QLabel(f'(Segment: '))
            self.ctrl_layout.addWidget(self.segment_selection_combo)
            self.ctrl_layout.addWidget(QLabel(f')'))
        self.ctrl_layout.addStretch()
        self.ctrl_layout.addWidget(self.enabled_checkbox)
        self.ctrl_layout.addWidget(self.remove_self_button)

        self._layout.addLayout(self.ctrl_layout)
        self.layout().addWidget(self.type_combo)

        self.edit_content = ReplaceableWidget()
        self.layout().addWidget(self.edit_content)

        self.type_choices = {
            'None': None_EditWidget(self.child_changed),
            'W Edit': W_EditWidget(self.child_changed),
            'S Edit': S_EditWidget(self.child_changed),
            'Styleclip Edit': StyleClip_EditWidget(self.child_changed),
        }
        for type in self.type_choices.keys():
            self.type_combo.addItem(type)

        self.type_combo.currentTextChanged.connect(
            lambda text: self._switch_mode(text)
        )
        self.segment_selection_combo.currentTextChanged.connect(
            lambda _: self.on_change()
        )

    def child_changed(self):
        self.on_change()

    def current_values(self):
        values = dotdict()
        values.type = self.type_combo.currentText()
        values.enabled = self.enabled_checkbox.isChecked()
        values.segment = self.segment_selection_combo.currentText()
        values.parameters = self.current_widget.current_values()
        return values


# Scrollable
class MultiEditWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)

        self._layout = QVBoxLayout()
        self._layout.setSpacing(0)
        self._layout.setContentsMargins(0,0,0,0) # Removes the useless space on the outside
        self.setLayout(self._layout)
        
        scroll_area = QScrollArea(self)
        self.layout().addWidget(scroll_area)

        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(200)

        scroll_content_container = QWidget()
        self.scroll_content = QVBoxLayout()
        self.scroll_content.setContentsMargins(3, 3, 3, 3)
        scroll_content_container.setLayout(self.scroll_content)
        scroll_area.setWidget(scroll_content_container)
        
        ext_btn = QToolButton()
        ext_btn.setIcon(self.style().standardIcon(QStyle.SP_ToolBarVerticalExtensionButton)) # QIcon.fromTheme("list-add")
        ext_btn.clicked.connect(lambda s: self._extend())

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(ext_btn)
        self.scroll_content.addLayout(btn_layout)
        self.scroll_content.addStretch()
        self.idx = 1

        self._children = {}
        self.push_update = None

    def gather_values(self):
        values = []
        for k, v in self._children.items():
            vals = v.current_values()
            if vals.type != 'None': # Skip None type for now.
                if vals.enabled:
                    values += [v.current_values()]
        return values

    def set_push_function(self, fn):
        self.push_update = fn

    def set_w_directions(self, direction_strings):
        # Currently will only update "at creation", meaning no live updates (not needed for now)
        _widget_params.w_edit.directions = direction_strings

    def set_s_presets(self, presets):
        # Preset is a {"name": (layer, channel)}
        _widget_params.s_edit.presets = {}
        for preset_name, preset_params in presets.items():
            _widget_params.s_edit.presets[preset_name] = dotdict({'layer': preset_params[0], 'channel': preset_params[1]})

    def set_s_limits(self, s_layer_limit, s_channel_limits):
        _widget_params.s_edit.limits = dotdict()
        _widget_params.s_edit.limits.layer_limit = s_layer_limit
        _widget_params.s_edit.limits.channel_limit = s_channel_limits

    def set_styleclip_models(self, models):
        _widget_params.styleclip_edit.models = models

    def set_segment_selection(self, choices):
        _widget_params.segment_selection = dotdict()
        _widget_params.segment_selection.choices = choices

    def _push_update(self):
        if self.push_update:
            values = self.gather_values()
            self.push_update(values)

    def _extend(self):
        new_child = self._make_editform()
        self.scroll_content.insertWidget(self.scroll_content.count() - 2, new_child)

    def _remove_editform(self, idx):
        assert idx in self._children, f"Requested removal of {idx}, not registered"
        child_to_remove = self._children[idx]
        assert child_to_remove, f"Requested removal of idx {idx}, widget not found"
        self.scroll_content.removeWidget(child_to_remove)
        del self._children[idx]
        child_to_remove.deleteLater()
        self.update()

    def _make_editform(self):
        idx = self.idx
        self.idx += 1

        def on_change():
            self._push_update()
        def on_remove():
            self._remove_editform(idx)
            self._push_update()
            
        new_child = EditWidget(idx, on_change, on_remove)
        self._children[idx] = new_child
        return new_child