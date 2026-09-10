import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

from hybrid_sample_generator.visualization.rendering import PanelSpec
from hybrid_sample_generator.visualization.widgets import ImageGrid


class VisualizerNavigationTests(unittest.TestCase):
    def setUp(self):
        self.grid = ImageGrid.__new__(ImageGrid)
        grid = self.grid
        grid.figure = Figure()
        grid.canvas = FigureCanvasAgg(grid.figure)
        grid.axes = tuple(grid.figure.subplots(1, 2))
        grid.cache = Mock()
        grid.cache.get.return_value = None
        grid.slice_index = 0
        grid.contrast = 1.0
        grid.channel = "auto"
        grid.show_mask = False
        grid.mask_opacity = 0.45
        grid.on_depth_changed = None
        grid._views = {}
        grid._full_views = {}
        grid._pan_axis = None
        grid._scroll_callback = Mock()
        image = np.arange(200, dtype=np.float32).reshape(1, 2, 10, 10)
        grid.set_specs((PanelSpec("First", image=image), PanelSpec("Second", image=image)))
        grid.canvas.draw()

    def _scroll(self, **kwargs):
        event = SimpleNamespace(
            inaxes=self.grid.axes[0], step=1, key=None, xdata=3.0, ydata=4.0,
        )
        event.__dict__.update(kwargs)
        self.grid._on_scroll(event)

    def test_zoom_is_cursor_centered_and_survives_display_changes(self):
        grid = self.grid
        axis, other = grid.axes
        before_x, before_y = axis.get_xlim(), axis.get_ylim()
        other_limits = other.get_xlim(), other.get_ylim()
        self._scroll()
        expected_x = [3 + (x - 3) / 1.2 for x in before_x]
        expected_y = [4 + (y - 4) / 1.2 for y in before_y]
        np.testing.assert_allclose(axis.get_xlim(), expected_x)
        np.testing.assert_allclose(axis.get_ylim(), expected_y)
        self.assertEqual((other.get_xlim(), other.get_ylim()), other_limits)
        self.assertGreater(axis.get_ylim()[0], axis.get_ylim()[1])
        grid.slice_index = 1
        grid.contrast = 2
        grid.render()
        np.testing.assert_allclose(axis.get_xlim(), expected_x)
        np.testing.assert_allclose(axis.get_ylim(), expected_y)
        grid.reset_view()
        self.assertEqual((axis.get_xlim(), axis.get_ylim()), (before_x, before_y))

    def test_shift_wheel_changes_slice_and_background_scroll_is_ignored(self):
        axis = self.grid.axes[0]
        limits = axis.get_xlim(), axis.get_ylim()
        self._scroll(key="shift", step=-1)
        self.grid._scroll_callback.assert_called_once_with(-1)
        self.assertEqual((axis.get_xlim(), axis.get_ylim()), limits)
        self._scroll(inaxes=None)
        self.assertEqual((axis.get_xlim(), axis.get_ylim()), limits)
        self._scroll()
        self.grid.set_specs(self.grid.specs)
        self.assertEqual((axis.get_xlim(), axis.get_ylim()), limits)

    def test_drag_pans_and_double_click_resets_selected_panel(self):
        grid = self.grid
        axis = grid.axes[0]
        self._scroll()
        before = axis.get_xlim()
        x, y = axis.transData.transform((4, 4))
        grid._on_press(SimpleNamespace(inaxes=axis, button=1, dblclick=False, x=x, y=y))
        grid._on_motion(SimpleNamespace(x=x + 20, y=y + 10))
        grid._on_release(None)
        self.assertNotEqual(axis.get_xlim(), before)
        self.assertAlmostEqual(axis.get_xlim()[1] - axis.get_xlim()[0], before[1] - before[0])
        panned = axis.get_xlim(), axis.get_ylim()
        grid.render()
        self.assertEqual((axis.get_xlim(), axis.get_ylim()), panned)
        grid._on_press(SimpleNamespace(inaxes=axis, button=1, dblclick=True))
        self.assertEqual((axis.get_xlim(), axis.get_ylim()), grid._full_views[axis])
